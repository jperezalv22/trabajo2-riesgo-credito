"""Une las secciones del reporte en un solo blog y lo convierte en pagina web.

    python construir_blog.py

1. Une reporte/secciones/*.md en orden y quita los comentarios <!-- --> (notas para el equipo)
   y los bloques "Referencias citadas en esta seccion": todas las referencias van juntas en la
   seccion 11.
2. Numera figuras y tablas en el orden en que aparecen. Cada seccion usa una numeracion
   provisional (Tabla 3, Tabla P3, Figura F1...) y aqui se cambia por la final, tanto en el
   titulo como en cada mencion del texto. Avisa si una figura o tabla no se cita en el texto
   (huerfana) o si se cita una que no existe.
3. Guarda reporte/blog.md y, con pandoc, docs/index.html (formulas con MathJax), copiando las
   figuras a docs/figuras/. GitHub Pages publica la carpeta docs/.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

from riesgo import configuracion

SECCIONES = Path("reporte/secciones")
BLOG = Path("reporte/blog.md")
SALIDA = Path("docs")
ESTILO = Path("reporte/estilo.html")

# Titulo de una figura o tabla: una linea que empieza con **Tabla P3.** o **Figura 2.**
TITULO = re.compile(r"^\*\*(Tabla|Figura) ([A-Z]?\d+)\.\*\*", re.MULTILINE)
# Una mencion: "Tabla P3", "Tablas 3 y 4", "Figuras P1 a P4", "Tablas 1, 2 y 3"
ID = r"[A-Z]?\d+"
MENCION = re.compile(rf"\b(Tablas?|Figuras?) ({ID}(?:(?:, | y | a ){ID})*)\b")
# notas para el equipo dentro de las secciones; no van en el blog
COMENTARIO = re.compile(r"<!--.*?-->\n?", re.DOTALL)
REFERENCIAS_SECCION = re.compile(r"^#### Referencias citadas en esta secci.n\n.*?(?=^#{1,3} |\Z)",
                                 re.MULTILINE | re.DOTALL)


def leer_secciones():
    archivos = sorted(SECCIONES.glob("[0-9][0-9]_*.md"))
    numeros = {int(a.name[:2]) for a in archivos}
    for faltante in sorted(set(range(1, max(numeros) + 1)) - numeros):
        print(f"AVISO: falta la seccion {faltante:02d}; el blog se arma sin ella.")
    return [(a.name, a.read_text(encoding="utf-8")) for a in archivos]


def quitar_referencias_de_seccion(texto):
    """Devuelve el texto sin los bloques de referencias y las referencias que tenian."""
    quitadas = []
    for bloque in REFERENCIAS_SECCION.findall(texto):
        quitadas += [l.strip() for l in bloque.splitlines()[1:] if l.strip()]
    return REFERENCIAS_SECCION.sub("", texto), quitadas


def poner_enlaces(texto, enlaces):
    """Cambia {{repositorio}}, {{app}}, {{blog}} y {{video}} por los enlaces de config.yaml."""
    for nombre, url in enlaces.items():
        texto = texto.replace("{{" + nombre + "}}", f"<{url}>" if url else "*(pendiente)*")
    return texto


def numeracion(texto):
    """{('Tabla', 'P3'): 15, ...} en el orden en que aparecen los titulos."""
    nueva, contador = {}, {"Tabla": 0, "Figura": 0}
    for tipo, provisional in TITULO.findall(texto):
        if (tipo, provisional) in nueva:
            sys.exit(f"ERROR: hay dos titulos '{tipo} {provisional}'. Cambie uno.")
        contador[tipo] += 1
        nueva[(tipo, provisional)] = contador[tipo]
    return nueva


def renumerar(texto, nueva):
    """Cambia cada numero provisional por el final, en titulos y menciones."""
    def cambiar(m):
        tipo = m.group(1).rstrip("s")
        ids = re.sub(ID, lambda i: str(nueva.get((tipo, i.group()), f"??{i.group()}")),
                     m.group(2))
        return f"{m.group(1)} {ids}"
    return MENCION.sub(cambiar, texto)


def revisar_citas(texto, nueva):
    """Avisa de figuras o tablas sin citar y de citas a figuras o tablas que no existen."""
    citas = {}
    for linea in texto.splitlines():
        es_titulo = TITULO.match(linea) or linea.startswith("![")
        for m in MENCION.finditer(linea):
            tipo = m.group(1).rstrip("s")
            for i in re.findall(ID, m.group(2)):
                citas.setdefault((tipo, i), 0)
                citas[(tipo, i)] += 0 if es_titulo else 1
    problemas = 0
    for clave in nueva:
        if citas.get(clave, 0) == 0:
            print(f"AVISO: {clave[0]} {clave[1]} no se cita en el texto (huerfana).")
            problemas += 1
    for clave in citas:
        if clave not in nueva:
            print(f"AVISO: se cita {clave[0]} {clave[1]}, pero no hay un titulo con ese numero.")
            problemas += 1
    return problemas


def revisar_referencias(quitadas, texto_referencias):
    """Las referencias de cada seccion deben estar en la lista final (seccion 11)."""
    faltan = 0
    for ref in sorted(set(quitadas)):
        inicio = ref[:40]
        if inicio not in texto_referencias:
            print(f"AVISO: la referencia '{inicio}...' no esta en 11_referencias.md")
            faltan += 1
    return faltan


def copiar_figuras(texto):
    """Copia cada imagen a docs/figuras/ y cambia la ruta en el texto."""
    carpeta = SALIDA / "figuras"
    carpeta.mkdir(parents=True, exist_ok=True)

    def cambiar(m):
        origen = (Path("reporte") / m.group(2)).resolve()
        destino = carpeta / f"{origen.parent.name}_{origen.name}"
        shutil.copyfile(origen, destino)
        return f"![{m.group(1)}](figuras/{destino.name})"
    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", cambiar, texto)


def a_html(texto, cfg):
    """Convierte el blog a HTML con pandoc. Si pandoc no esta, avisa y no se cae."""
    if shutil.which("pandoc") is None:
        print("AVISO: pandoc no esta instalado; se guardo reporte/blog.md pero no docs/index.html."
              " Instale pandoc (https://pandoc.org/installing.html) y corra de nuevo.")
        return False
    b = cfg["blog"]
    temporal = SALIDA / "blog_web.md"
    temporal.write_text(copiar_figuras(texto), encoding="utf-8")
    subprocess.run(["pandoc", str(temporal), "-o", str(SALIDA / "index.html"), "--standalone",
                    "--mathjax", "--toc", "--toc-depth=2", "-M", f"title={b['titulo']}",
                    "-M", f"subtitle={b['subtitulo']}", "-M", "lang=es",
                    "-M", "toc-title=Índice", "-M", f"date={b['fecha']}",
                    *[a for autor in b["autores"] for a in ("-M", f"author={autor}")],
                    f"--include-in-header={ESTILO}"], check=True)
    temporal.unlink()
    # sin esto GitHub Pages pasa la carpeta por Jekyll
    (SALIDA / ".nojekyll").write_text("", encoding="utf-8")
    return True


def main():
    a = configuracion.argumentos(__doc__).parse_args()
    cfg = configuracion.cargar(a.config, a.cambios)

    partes, quitadas, referencias = [], [], ""
    for nombre, texto in leer_secciones():
        if nombre.startswith("11_"):
            referencias = texto
        texto, refs = quitar_referencias_de_seccion(COMENTARIO.sub("", texto))
        partes.append(texto.strip())
        quitadas += refs
    texto = poner_enlaces("\n\n".join(partes) + "\n", cfg["app"]["enlaces"])

    nueva = numeracion(texto)
    texto = renumerar(texto, nueva)
    problemas = revisar_citas(texto, {(t, str(n)): n for (t, _), n in nueva.items()})
    problemas += revisar_referencias(quitadas, referencias)

    BLOG.write_text(texto, encoding="utf-8")
    hecho = a_html(texto, cfg)
    tablas = sum(1 for t, _ in nueva if t == "Tabla")
    print(f"Blog: {len(partes)} secciones, {tablas} tablas, {len(nueva) - tablas} figuras, "
          f"{problemas} avisos. Guardado en {BLOG}" + (" y docs/index.html" if hecho else ""))


if __name__ == "__main__":
    main()
