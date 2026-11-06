# Referencias como las recordaba la IA, ANTES de verificarlas

Escrito por Claude Code (parte C) el 2026-11-05, antes de consultar Crossref, doi.org o las
páginas de las revistas. Sirve de evidencia para la cacería: se compara con lo verificado en
`reporte/verificacion/referencias_verificadas.md`. No se debe editar después de verificar.

1. McCulloch, W. S., y Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity. Bulletin of Mathematical Biophysics, 5(4), 115-133. DOI 10.1007/BF02478259
2. Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. Psychological Review, 65(6), 386-408. DOI 10.1037/h0042519
3. Rumelhart, D. E., Hinton, G. E., y Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323(6088), 533-536. DOI 10.1038/323533a0
4. Hornik, K., Stinchcombe, M., y White, H. (1989). Multilayer feedforward networks are universal approximators. Neural Networks, 2(5), 359-366. DOI 10.1016/0893-6080(89)90020-8
5. Nair, V., y Hinton, G. E. (2010). Rectified linear units improve restricted Boltzmann machines. ICML 2010, 807-814.
6. Goodfellow, I., Bengio, Y., y Courville, A. (2016). Deep learning. MIT Press.
7. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., y Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. JMLR, 15(56), 1929-1958.
8. Prechelt, L. (1998). Early stopping - but when? En G. B. Orr y K.-R. Müller (Eds.), Neural Networks: Tricks of the Trade (LNCS 1524, pp. 55-69). Springer. DOI 10.1007/3-540-49430-8_3
9. Loshchilov, I., y Hutter, F. (2019). Decoupled weight decay regularization. ICLR 2019. arXiv 1711.05101
10. Hanley, J. A., y McNeil, B. J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. Radiology, 143(1), 29-36. DOI 10.1148/radiology.143.1.7063747
11. Massey, F. J. (1951). The Kolmogorov-Smirnov test for goodness of fit. JASA, 46(253), 68-78. DOI 10.1080/01621459.1951.10500769
12. Hand, D. J., y Henley, W. E. (1997). Statistical classification methods in consumer credit scoring: A review. JRSS Series A, 160(3), 523-541. DOI 10.1111/j.1467-985X.1997.00078.x
13. Lessmann, S., Baesens, B., Seow, H.-V., y Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. EJOR, 247(1), 124-136. DOI 10.1016/j.ejor.2015.05.030
14. Grinsztajn, L., Oyallon, E., y Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? NeurIPS 35 (Datasets and Benchmarks). arXiv 2207.08815
15. Shwartz-Ziv, R., y Armon, A. (2022). Tabular data: Deep learning is not all you need. Information Fusion, 81, 84-90. DOI 10.1016/j.inffus.2021.11.011
16. Basel Committee on Banking Supervision (2006). International convergence of capital measurement and capital standards: A revised framework (comprehensive version). BIS.

Fórmulas recordadas que también se van a comprobar:
- RMSprop en PyTorch: v = alpha v + (1 - alpha) g^2, con alpha = 0.99 por defecto (en las diapositivas de Hinton, 0.9).
- Adam en PyTorch: theta -= lr * m_hat / (sqrt(v_hat) + eps), eps = 1e-8, betas (0.9, 0.999).
- nn.Dropout: en entrenamiento multiplica por 1/(1 - p) lo que no apaga; en evaluación no hace nada.
- Derivada de ReLU en 0 en PyTorch: 0.
