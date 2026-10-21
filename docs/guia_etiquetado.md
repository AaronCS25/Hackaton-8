# Guía de etiquetado

## Contexto de la tarea

El router recibe un mensaje de un cliente de una agencia peruana de viajes y eventos y devuelve un JSON que indica la intención del mensaje, el agente especializado que debe atenderlo, la prioridad, si la acción requiere aprobación humana y los datos clave del pedido. Los mensajes llegan por WhatsApp o chat web, por lo que suelen ser informales y pueden contener errores de tipeo, jerga peruana, montos escritos en palabras y más de un pedido a la vez.

El etiquetado se divide en dos niveles. La persona que etiqueta define la intención, los atributos factuales y los slots. Las etiquetas `agente`, `prioridad` y `requiere_aprobacion_humana` se calculan con reglas deterministas implementadas en `src/router/esquema.py`, lo que elimina la variabilidad entre etiquetadores en esos campos.

## Formato de salida

```json
{"intencion": "modificar", "agente": "logistica", "prioridad": "P3", "requiere_aprobacion_humana": false, "slots": {"destino": "Cusco", "fecha": "15 de noviembre", "n_personas": 4, "monto": null}}
```

## Intención

| Intención | Definición |
|---|---|
| `cotizar` | El cliente pide precio o presupuesto de un servicio que todavía no reservó. |
| `reservar` | El cliente quiere confirmar o separar un servicio. |
| `modificar` | El cliente pide cambiar fecha, número de personas, destino, horario o pasajeros de una reserva existente. |
| `cancelar` | El cliente pide anular una reserva existente, total o parcialmente. |
| `reclamo` | El cliente se queja de un servicio recibido o en curso, sin pedir devolución de dinero y sin riesgo para personas. |
| `pago_reembolso` | El mensaje trata de pagos, cuotas, comprobantes, cobros duplicados o devoluciones de dinero. |
| `emergencia_en_viaje` | Ocurre un incidente durante un viaje en curso que afecta la seguridad o la continuidad del servicio. |
| `consulta_info` | El cliente pide información general como requisitos, documentos, clima, equipaje, horarios o medios de pago. |

Cuando un mensaje contiene varios pedidos, se etiqueta la intención de mayor precedencia según el orden `emergencia_en_viaje`, `cancelar`, `pago_reembolso`, `modificar`, `reclamo`, `reservar`, `cotizar`, `consulta_info`. Una queja que además pide devolución de dinero se etiqueta como `pago_reembolso`.

## Atributos factuales

| Atributo | Tipo | Valor verdadero cuando |
|---|---|---|
| `riesgo_personas` | bool | Existe riesgo para la salud o integridad física de alguna persona, como accidente, enfermedad, robo con violencia, extravío de un menor o desastre natural. Solo aplica a `emergencia_en_viaje`. |
| `plazo` | categoría | Ver la tabla de plazos. |
| `dinero_en_disputa` | bool | Hay un cobro ya realizado que el cliente objeta, como un cobro duplicado, un reembolso pendiente o un cargo no reconocido. |
| `mueve_dinero` | bool | Atender el pedido implica ejecutar un cobro, un pago, un reembolso o una devolución. |
| `pide_humano_o_legal` | bool | El cliente pide hablar con una persona o menciona Indecopi, el libro de reclamaciones, una denuncia o un abogado. |
| `datos_personales` | bool | El pedido implica registrar, corregir o transferir datos personales, como DNI, pasaporte, titular de la reserva o datos de pasajeros. |
| `excepcion` | bool | El cliente pide una excepción a la política comercial, como un descuento especial, la exoneración de una penalidad o una prórroga de pago. |
| `tema_pagos` | bool | Una `consulta_info` trata de medios de pago, cuotas o facturación. |

| Plazo | Criterio |
|---|---|
| `en_curso` | El viaje o evento ya empezó. |
| `menos_24h` | El servicio ocurre hoy o mañana. |
| `hasta_72h` | El servicio ocurre en los próximos tres días. |
| `mas_72h` | El mensaje menciona una fecha concreta posterior a tres días. |
| `sin_fecha` | El mensaje no permite ubicar el servicio en el tiempo. |

## Reglas derivadas

El agente se asigna con la primera regla que se cumple.

| Orden | Condición | Agente |
|---|---|---|
| 1 | `riesgo_personas` | `seguridad` |
| 2 | `pide_humano_o_legal` o intención `reclamo` | `humano` |
| 3 | Intención `pago_reembolso` | `finanzas` |
| 4 | Intención `consulta_info` con `tema_pagos` | `finanzas` |
| 5 | Intención `consulta_info` sin `tema_pagos` | `clima_info` |
| 6 | Cualquier otro caso | `logistica` |

La prioridad también se asigna con la primera regla que se cumple.

| Orden | Condición | Prioridad |
|---|---|---|
| 1 | `riesgo_personas` o intención `emergencia_en_viaje` | `P1` |
| 2 | Plazo `en_curso`, `menos_24h` o `hasta_72h`, o `dinero_en_disputa`, o `pide_humano_o_legal` | `P2` |
| 3 | Plazo `mas_72h` | `P3` |
| 4 | Plazo `sin_fecha` | `P4` |

La acción requiere aprobación humana cuando la intención es `cancelar` o cuando se cumple alguno de los atributos `mueve_dinero`, `datos_personales`, `excepcion` o `riesgo_personas`. Estas acciones son irreversibles, comprometen dinero o datos personales, o activan un protocolo de emergencia, por lo que un agente automático no debe ejecutarlas sin supervisión.

## Slots

| Slot | Tipo | Criterio |
|---|---|---|
| `destino` | texto o null | Lugar principal del servicio, escrito con su nombre propio estándar, por ejemplo "Cusco" o "Punta Cana". |
| `fecha` | texto o null | Fecha o referencia temporal tal como aparece en el mensaje, en minúsculas y sin el artículo inicial, por ejemplo "15 de noviembre", "este sábado" o "mañana". |
| `n_personas` | entero o null | Número total de personas del servicio. Una expresión como "somos 3 más mi mamá" se registra como 4. |
| `monto` | número o null | Monto de dinero principal del mensaje, ya sea presupuesto, cobro o reembolso, convertido a número. Una expresión como "4 lucas y media" se registra como 4500. |

Un slot que el mensaje no menciona se registra como null. El modelo no debe inferir valores ausentes.
