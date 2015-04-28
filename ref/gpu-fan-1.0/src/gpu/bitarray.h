#ifndef BIT_ARRAY_H
#define BIT_ARRAY_H

#ifndef INT_BIT
#define INT_BIT 32
#endif
/* array index for character containing bit */
#define BIT_INT(bit)         ((bit) / INT_BIT)
/* position of bit within character */
#define BIT_IN_INT(bit)      (1 << (INT_BIT - 1 - ((bit)  % INT_BIT)))

#define BITS_TO_INTS(bits)   ((((bits) - 1) / INT_BIT) + 1)  
#endif  /* BIT_ARRAY_H */
