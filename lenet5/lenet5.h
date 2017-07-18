/*
 * lenet5.h
 *
 *  Created on: 2017. 5. 22.
 *      Author: woobes
 */

#ifndef SRC_LENET5_LENET5_H_
#define SRC_LENET5_LENET5_H_


#include "classify_lib.h"
#ifdef HW_TEST
#include "hw_layers/image_convolution.h"
#include "hw_layers/image_fullyconnected.h"
#include "hw_layers/image_pool.h"
#else
#include "sw_layers/image_convolution_sw.h"
#endif
#include "sw_layers/image_fullyconnected_sw.h"
#include "sw_layers/image_pool_sw.h"


#endif /* SRC_LENET5_LENET5_H_ */
