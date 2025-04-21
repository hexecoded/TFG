# Prognosis y detección del cáncer de piel usando Deep Learning
![License](https://img.shields.io/badge/license-GNU%20GPL%20v3-orange.svg?style=for-the-badge)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
<p align="center" width="100%%">
    <img width="20%" src="https://github.com/hexecoded/TFG/blob/isic-pytorch/Memoria/imagenes/logo.png">
</p>
<p>
TFG realizado para la finalización del grado de Ingeniería Informática en la Universidad de Granada (2024). Su desarrollo se basa en la construcción de un modelo neuronal profundo capaz de ser ejecutado en un dispositivo móvil de forma eficiente, empleando para ello las nuevas funcionalidades de Pytorch y el proceso algorítmico cuantización de modelos.
  
## Descripción

Las lesiones en la piel son muy comunes en nuestro día a día. Puede tratarse desde una simple abrasión hasta lunares de extraño aspecto. Debido a la alta exposición a la radiación solar que recibimos por el deterioro de la capa de ozono, las lesiones son cada vez más comunes si no usamos protección. Si las quemaduras se producen de forma continuada, pueden surgir problemas como el cáncer de piel, cuya probabilidad va en aumento, siendo un 2.6% la probabilidad de sufrirlo en algún momento vital. Sólo en España, fallecieron casi 1100 personas en 2022.
</p>

Normalmente, este tipo de lesiones no son fáciles de analizar a simple vista en sus fases iniciales. Por ello, se quiere construir un modelo capaz de identificar los diferentes tipos de lesiones y facilitar el autodiagnóstico de lesiones cancerosas, con el fin de acelerar el tratamiento lo máximo posible y reducir la invasividad del tumor. Sólo en el 11% de los casos, se identifican los tumores terminales a tiempo.
Dado a que en la actualidad son frecuentes los dispositivos móviles, el objetivo consistirá en realizar una aplicación gratuita que permita utilizar el modelo diseñado para hacer conocer al usuario un resultado preliminar. Usando técnicas de visión por computación y aprendizaje profundo, se realizará el tratamiento de la imagen tomada por el usuario en su lesión, y se realizará su diagnóstico basado en los casos estudiados por el modelo.

## Justificación

Debido a la necesidad de facilitar el acceso al diagnóstico de forma orientativa, sin necesidad  de medios tecnológicos especializados, surge la necesidad de algún medio de diagnóstico gratuito capaz de acelerar el proceso mediante la prognosis, y reducir la tasa de mortalidad de la enfermedad. Encontramos, de esta forma, un área de conocimiento poco explorada: la creación de una aplicación móvil, capaz de emplear modelos detección y prognosis de cáncer de piel, empleando mecanismos específicos para dispositivos de baja potencia.

Los smartphones, al tratarse de un dispositivo altamente extendidos en la sociedad, pueden servir como un medio clave a la hora de extender modelos de salud críticos como el de diagnóstico de enfermedades cancerosas. De esta forma, podríamos acortar los largos tiempos de espera, realizando un diagnóstico previo para declarar cuáles son los casos que requieren mayor prioridad médica.

Sin embargo, no existen modelos de este ámbito lo suficientemente ligeros como para ser portados de forma local dentro del propio terminal, que no necesiten de conexión a Internet, y que velen por la seguridad y privacidad de los datos. Esto convierte el problema en una interesante temática de investigación, la cual será tratada en este documento.

## Objetivos
El uso de un modelo de aprendizaje profundo en dispositivos de potencia reducida permite:
- Facilitar la toma de decisiones de los expertos dermatólogos en casos complicados, a modo de sistema de ayuda a la toma de decisiones.
- Defender la utilidad de los modelos de aprendizaje profundo en el estudio y evaluación de casos en el ámbito médico, siendo en este caso la identificación de la patología a nivel macroscópico, sin necesidad de realizar una biopsia con el posible riesgo que esto conlleva.
- Crear una base de datos robusta y completa capaz de representar adecuadamente la población real de la enfermedad, partiendo, para ello, de la construcción de un conjunto de datos mediante una técnica no explotada en el estado del arte: la fusión de conjuntos de imágenes.
- Realizar, mediante un enfoque novedoso, el entrenamiento de un modelo con dos niveles de modelos especializados: uso de clasificación binaria para prediagnóstico de la enfermedad, y especificación de la misma mediante modelos especializados separados para enfermedades benignas y malignas, empleando la cuantización como método de simplificación.
- Mejorar la accesibilidad, al hacer uso de un dispositivo esencial en nuestra vida diaria como por ejemplo, los teléfonos móviles, aprovechando la posibilidad de que la manifestación de los tumores de piel son visibles a nivel macroscópico.
- Mostrar el desempeño de los modelos cuantizados y simplificados mediante el uso de una aplicación para dispositivos Android, capaz de ejecutarlos y mostrar su funcionamiento.

## Contenido
En este repositorio, podrá encontrar toda la documentación asociada al estudio del estado del arte, el análisis de las posibles alternativas y enfoque seguido a la hora de realizar el modelo definitivo, así como el diseño de la aplicación Android capaz de ejecutar con imágenes tomadas por la cámara para analizarlas en un breve período de tiempo.

La implementación propiamente dicha de la aplicación puede encontrarse en el siguiente repositorio: [TFG (Android)](https://github.com/hexecoded/TFG-Android)
