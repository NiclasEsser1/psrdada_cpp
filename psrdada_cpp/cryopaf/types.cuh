/*
 * cryopaf_constants.hpp
 *
 *  Created on: Aug 3, 2020
 *      Author: Niclas Esser
 */

#ifndef CRYOPAF_CONSTANTS_H_
#define CRYOPAF_CONSTANTS_H_

namespace psrdada_cpp{
namespace cryopaf{

#define NSAMPLE 333
#define NCHANNEL 64
#define NANTENNA 1
#define NPOL 2
#define NBEAM 2
#define NTHREAD 1024
#define WARP_SIZE 32

// Enumeration for beamformer type
enum{SIMPLE_BF_TAFPT, BF_TFAP};

struct bf_config_t{
   std::size_t n_samples;
   std::size_t n_channel;
   std::size_t n_antenna;
   std::size_t n_pol;
   std::size_t n_beam;
   std::size_t n_thread;
   std::size_t n_warp_size;
   std::size_t bf_type;
};

}
}

#endif /* CRYOPAF_CONSTANTS_H_ */
