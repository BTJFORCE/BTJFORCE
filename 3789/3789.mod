FILE theView mod 1.00

##ROTATING-OBJECT
##	NAME  object_name
##	OFFSET x y z
##	SHAFT  x y z i j k
##	MIN   min_angle (deg)
##	MAX   max_angle (deg)
##	[ANIMATION  CONTINUOUS | BOUNCING ]
##	SPEED   angle (deg/min)
##END-OBJECT

##TRANSLATING-OBJECT
##	NAME  object_name
##	OFFSET x y z
##	SLEDGE x1 y1 z1 x2 y2 z2
##	MIN    d1 
##	MAX    d2
##	[ANIMATION BOUNCING | CONTINUOUS]
##	SPEED  d
##END-OBJECT

ROTATING-OBJECT
  NAME bowport
  OFFSET 0.0 73.0 11.255
  SHAFT  100 0 5 1 0 0
  MIN    0
  MAX    80
  SPEED 120
END-OBJECT



END-FILE
