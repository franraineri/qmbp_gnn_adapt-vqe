# Classical representability — resumen

Tolerancia de convergencia (|ΔE| ≤): 1e-06

chi necesario para converger la energía, por método/N/h.

Un guion (—) = no converge dentro de los chi probados.


## mps_dmrg

### chain_1d

Celda = chi necesario (S_vN máx en paréntesis).

| N \ h | h=0.5 | h=1 | h=2 |
|---|---|---|---|
| 4 | 2 (S=0.60) | 2 (S=0.28) | 2 (S=0.09) |
| 10 | 4 (S=0.69) | 8 (S=0.38) | 4 (S=0.09) |


## Veredicto de viabilidad

El chi necesario crece con la entropía de entrelazamiento S. Estados con S baja (lejos del punto crítico) son representables con chi chico → viable. S alta (crítico y/o N grande) empuja chi hacia valores impracticables → el método clásico deja de escalar. Los guiones y los chi grandes marcan esa frontera.
