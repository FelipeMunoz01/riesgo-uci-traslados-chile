# Registro de decisiones

Cada decisión de este proyecto se tomó midiendo, no por intuición. Este documento
guarda los números que respaldan cada una, incluidas las que resultaron al revés de
lo esperado.

Todos los experimentos evalúan sobre el mismo conjunto de prueba: el año **2024**
completo, no visto durante el entrenamiento.

---

## 1. Tres variables filtraban información del futuro

El GRD es una base de **egresos**: la ficha se llena al alta. Una columna solo sirve
como predictor si su *valor* se conocía en el momento de decidir, no solo si existe
en el archivo.

| variable | evidencia | veredicto |
|---|---|---|
| `especialidad_medica` | `MEDICINA INTENSIVA ADULTO` tiene 98,0 % de tasa de UCI sobre 64.420 casos. Era la variable con mayor importancia del modelo. | fuera |
| `n_diagnosticos_secundarios` | Con 0 diagnósticos la tasa es 3,1 %; con 15 o más, 54,0 %. Cuenta casillas llenadas durante toda la estadía. | fuera |
| `n_procedimientos` | 1 procedimiento, 2,9 %; 10 o más, 30,3 %. Además arrastraba procedimientos de soporte vital hechos dentro de la UCI. | fuera |
| `procedimiento_principal` | Se conoce exacto si la cirugía es programada, y como intención si es de urgencia. | se queda |

Ablación con todas las features contra solo las conocibles al ingreso:

| escenario | ROC-AUC | AP |
|---|---|---|
| todas las features | 0,955 | 0,818 |
| sin `especialidad_medica` | 0,945 | 0,773 |
| sin `n_procedimientos` | 0,949 | 0,802 |
| sin las cuatro | 0,912 | 0,668 |

La caída del ROC-AUC parece moderada, pero la Average Precision baja un 18 % relativo,
y esa es la métrica que importa cuando hay que priorizar el tramo de mayor riesgo.

**Por qué no basta con decir que el número estaba inflado:** el modelo original *no se
podía ejecutar*. En producción, al momento de decidir, esos valores no existen. Un
mismo paciente con neumonía daba 9,9 % con lo conocible al ingreso y 83,0 % con la
información codificada al alta.

---

## 2. El índice de Charlson estaba roto, y arreglarlo no sirvió

Los diagnósticos vienen con punto (`E11.5`), pero el índice estaba escrito sin punto
(`E115`). Resultado: **85 de 177 prefijos nunca podían coincidir**. Tres categorías
quedaban completamente muertas, entre ellas diabetes con complicación.

Se arregló y se midió:

| variante | ROC-AUC | AP |
|---|---|---|
| base, sin medida de comorbilidad | 0,9190 | 0,7083 |
| base + Charlson roto | 0,9208 | 0,7116 |
| base + Charlson arreglado | 0,9213 | 0,7127 |
| base + CC/MCC | 0,9320 | 0,7373 |
| base + CC/MCC + Charlson arreglado | 0,9325 | 0,7385 |

Arreglar los 85 prefijos aportó **+0,001 de AP**. Sobre CC/MCC, Charlson aporta +0,0012,
que es ruido. La clasificación CC/MCC del IR-GRD chileno hace el trabajo con siete veces
más señal.

**Decisión: eliminar Charlson en vez de arreglarlo.** Menos código, un tercio menos de
tiempo de extracción y un campo menos que pedirle al usuario.

---

## 3. La población incluía atenciones que no son hospitalizaciones

| exclusión | n | tasa de UCI |
|---|---|---|
| Cirugía mayor ambulatoria | 889.832 | 1,2 % |
| Sesiones de quimioterapia (`Z51`) | 100.432 | 3,7 % |
| Sesiones de diálisis (`Z49`) | 22.487 | 1,3 % |

Nunca son objeto de una decisión de traslado a cama crítica, y como negativos fáciles
inflaban la métrica global.

**El corte va por diagnóstico, no por procedimiento.** El procedimiento `39.95`
(hemodiálisis) tiene 19,4 % de tasa de UCI, porque también se dializa a pacientes
críticos con falla renal aguda. Cortar por procedimiento habría borrado pacientes reales.

`TIPO_ACTIVIDAD` tampoco servía: la categoría `HOSPITALIZACIÓN DIURNA` solo existe en
2019, así que excluirla habría sido inconsistente entre años.

Medido sobre la misma población en los dos casos:

| variante | ROC-AUC | AP |
|---|---|---|
| con ambulatorios, sin la columna | 0,9121 | 0,7248 |
| con ambulatorios + `tipo_actividad` como feature | 0,9130 | 0,7275 |
| sin ambulatorios | 0,9140 | 0,7299 |
| sin ambulatorios + `tipo_actividad` como feature | 0,9142 | 0,7308 |

Se descartó usar `tipo_actividad` como feature: +0,0009 no justifica una columna cuyo
esquema cambió entre años.

---

## 4. Menos datos, pero más recientes, rinde mejor

2019 subregistraba la etiqueta: 9,1 % contra ~14 % en los demás años. El ingreso directo
a servicio crítico era 6,2 % contra 10-12 %, y `SERVICIOTRASLADO1` estaba poblado en
14,4 % de los casos contra 20,6 %. No es efecto de la pandemia: 2023 también está en 14 %.

| ventana de entrenamiento | filas | ROC-AUC | AP |
|---|---|---|---|
| 2019-2023 | 3.875.015 | 0,9131 | 0,7311 |
| **2020-2023** | **2.929.741** | **0,9146** | **0,7358** |
| 2022-2023 | 1.561.631 | 0,9161 | 0,7395 |

Patrón monótono e idéntico en los dos modelos. Se eligió 2020-2023 y no 2022-2023, que
mide mejor, para conservar volumen en las categorías poco frecuentes.

---

## 5. La etiqueta medía racionamiento, no gravedad

Al validar apareció algo contraintuitivo: el riesgo **bajaba** con la edad. Contrastar
el uso de UCI contra la mortalidad explicó por qué.

**Neumonía (J18):**

| edad | uso de UCI/UTI | mortalidad | razón muerte/UCI |
|---|---|---|---|
| menos de 50 | 22,3 % | 3,9 % | 0,18 |
| 50 a 64 | 24,9 % | 12,0 % | 0,48 |
| 65 a 79 | 19,2 % | 17,6 % | 0,92 |
| **80 o más** | **8,6 %** | **23,3 %** | **2,71** |

Los mayores de 80 usan 2,6 veces menos UCI y mueren 6 veces más. Mismo patrón en sepsis:
el uso baja de 55,4 % a 26,3 % mientras la mortalidad sube de 18,3 % a 50,3 %.

No es que estén menos graves. Es limitación del esfuerzo terapéutico y racionamiento de
camas. Un modelo entrenado solo sobre "¿ocupó cama crítica?" diría **riesgo bajo** justo
para los pacientes más frágiles.

Etiqueta compuesta, `UCI/UTI o fallecimiento`:

| | etiqueta original | solo mortalidad | compuesta |
|---|---|---|---|
| Neumonía, menos de 50 | 22,3 % | 3,9 % | 24,1 % |
| Neumonía, 80 o más | **8,6 %** | 23,3 % | **28,8 %** |
| Sepsis, menos de 50 | 55,4 % | 18,3 % | 60,2 % |
| Sepsis, 80 o más | **26,3 %** | 50,3 % | **63,5 %** |

Corrige la inversión perversa. No la vuelve monótona con la edad, y probablemente está
bien que no lo haga: en insuficiencia cardíaca el riesgo sigue bajando con la edad incluso
con la compuesta, lo que apunta a señal clínica real, ya que el paciente joven con ICC
suele ser miocardiopatía o candidato a trasplante.

**Decisión: dos modelos en paralelo, no reemplazar la etiqueta.** Responden preguntas
distintas y el comité necesita ambas. Verificado con un paciente tipo:

| edad | cama crítica | riesgo clínico | divergencia |
|---|---|---|---|
| 40 | 7,3 % | 11,6 % | +4,3 pp |
| 70 | 4,8 % | 11,6 % | +6,8 pp |
| 90 | **3,4 %** | **13,6 %** | **+10,3 pp** |

Cuando los dos puntajes divergen, esa divergencia es la señal más útil. La aplicación
marca los casos con más de 5 puntos de diferencia.

**Distinción clave:** `tipo_alta` es fuga como *feature* pero es válida como *etiqueta*.
Las etiquetas pueden ser post-hoc; es lo que se quiere predecir. Solo las features tienen
que conocerse al momento de decidir.

---

## 6. El diagnóstico necesita dos niveles de detalle

La categoría de 3 caracteres agrupa cuadros muy distintos:

| código | n | tasa de UCI |
|---|---|---|
| I21.0, infarto de pared anterior | 10.364 | 76,9 % |
| I21.4, infarto sin elevación del ST | 19.833 | 52,9 % |

24 puntos de diferencia invisibles para el modelo. Pero usar solo el código completo
empeora: hay 8.947 distintos y el top-200 cubre 58 % de los casos, contra 80 % de la
categoría.

| variante | AP cama | AP clínico |
|---|---|---|
| solo categoría | 0,7358 | 0,7396 |
| **categoría + subcódigo** | **0,7406** | **0,7433** |
| solo subcódigo | 0,7172 | 0,7202 |

**Decisión: los dos a la vez.** La categoría da cobertura amplia, el subcódigo da detalle
donde hay volumen.

---

## 7. Un bug real que el modelo compensaba solo

`date_diff('year', ...)` en DuckDB cuenta cruces de 1 de enero, no cumpleaños. Un recién
nacido del 31 de diciembre ingresado el 1 de enero figuraba con 1 año. Afectaba a 49.081
casos, y el tramo de 0 años tiene 47,4 % de tasa de UCI contra 18,6 % al año.

Se corrigió a años cumplidos. Efecto en la métrica, midiendo el mismo conjunto de features
sobre los datos antiguos y los corregidos: **−0,0002 de ROC-AUC**. Es decir, ninguno.

El árbol ya identificaba a los recién nacidos por otra vía, porque los diagnósticos
neonatales son muy distintivos. Pero la corrección importa igual: en la aplicación **el
usuario escribe la edad**, y si el modelo se entrenó con una codificación corrida, quien
escribe 0 recibe una predicción construida sobre un grupo mal etiquetado.

Vale como recordatorio: que una métrica agregada no se mueva no significa que un bug sea
inofensivo.

---

## 8. Un modelo por patología solo gana donde el global es débil

| patología | AUC global | AUC especializado | diferencia |
|---|---|---|---|
| ACV isquémico (I63) | 0,791 | 0,823 | **+0,032** |
| Infarto agudo (I21) | 0,793 | 0,813 | **+0,020** |
| Insuficiencia cardíaca (I50) | 0,799 | 0,809 | +0,011 |
| Neumonía bacteriana (J15) | 0,867 | 0,868 | +0,002 |
| Neumonía (J18) | 0,870 | 0,869 | −0,001 |
| Fractura de cadera (S72) | 0,821 | 0,817 | −0,004 |

Donde el modelo global ya andaba bien, especializar no aporta o resta: pierde los patrones
compartidos entre patologías a cambio de menos datos.

Y el techo es bajo por una razón de fondo: acotar a una patología **no agrega variables**,
solo reduce filas. El salto real exigiría datos clínicos que el GRD no tiene, como escala
de Killip, troponina, NIHSS o tiempo desde el inicio de los síntomas.

**Decisión: no hacerlo.** Descartado por relación beneficio/complejidad.

---

## Resumen del recorrido

| paso | ROC-AUC |
|---|---|
| versión inicial, con fuga de datos | 0,955 |
| quitar las variables contaminadas | 0,931 |
| quitar la población ambulatoria | 0,913 |
| corregir bugs (edad, Charlson, reescritura de comorbilidades) | 0,913 |
| excluir 2019 | 0,915 |
| subcódigo CIE-10 y 19 comorbilidades | **0,918** |

De todo el trabajo, las únicas ganancias netas de métrica fueron excluir 2019 y agregar
el subcódigo, unos +0,005 de AP cada una. Los arreglos de correctitud fueron neutros en
capacidad predictiva.

Lo que cambió no fue la precisión, sino la validez: el 0,918 describe algo que va a
ocurrir de verdad, y el 0,955 describía un escenario imposible.
