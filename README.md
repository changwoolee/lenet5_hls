lenet5_test
===========

TO-DO
-----


1. Implement C3, S1, S2 layers


* CONVOLUTION_LAYER_1

LUT usage : 67% -> 28%

sw cycle : 1511361
hw cycle :  341729
speed up : x4.42

* CONVOLUTION_LAYER_2

sw cycle : 1807944
hw cycle :  200834
speed up : x9.00

* CONVOLUTION_LAYER_3
not implemented yet


* Performance Log

| Date |             |     | 2017-05-26             |        |             | 2017-05-27            |         |             |           |
|------|-------------|-----|------------------------|--------|-------------|-----------------------|---------|-------------|-----------|
| Mode | SW only -O3 |     | release (HW : C1 only) |        |             | release(HW : C1 & C3) |         |             |           | 
|      | w/ no tanh  |     | SW                     | HW     | Speed up    | SW                    | HW      | Speed up    | Freq(MHz) | 
| C1   | 689365      | 28% | 1511349                | 311203 | 4.856473106 | 1511361               | 341729  | 4.422688739 | 100       | 
| S2   | 62075       | 3%  |                        |        | #DIV/0!     |                       |         | #DIV/0!     |           | 
| C3   | 1298909     | 54% |                        |        | #DIV/0!     | 1807944               | 200834  | 9.002180906 | 142.86    | 
| S4   | 20409       | 1%  |                        |        | #DIV/0!     |                       |         | #DIV/0!     |           | 
| C5   | 263015      | 11% |                        |        | #DIV/0!     |                       |         | #DIV/0!     |           | 
| Tot  | 2423750     |     |                        |        |             | 3783112               | 1016737 | 3.720836362 |           | 
