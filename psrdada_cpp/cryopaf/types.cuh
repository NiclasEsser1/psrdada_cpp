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

#define NTHREAD 1024
#define WARP_SIZE 32
#define WARPS NTHREAD/WARP_SIZE
#define SHARED_IDATA (NTHREAD*2)


#define N_THREAD_CB 256
#define WARPS_CB (N_THREAD_CB/WARP_SIZE)
#define N_POL_CB 2
#define N_ELEMENTS_CB 32
#define N_CHANNEL_CB 32
#define N_BEAMS_CB 512
#define INTERVAL_CB 64
#define N_TIMESTAMPS_CB 4096
#define N_TIMESTAMPS_OUT_CB (INTERVAL_CB/N_TIMESTAMPS_CB)

// Enumeration for beamformer type
enum{SIMPLE_BF_TAFPT, BF_TFAP, BF_TFAP_TEX, CUTENSOR_BF_TFAP, BF_TFAP_V2};

struct bf_config_t{
   std::size_t n_samples;
   std::size_t n_channel;
   std::size_t n_antenna;
   std::size_t n_pol;
   std::size_t n_beam;
   std::size_t interval;
   std::size_t bf_type;
};

}
}



#endif /* CRYOPAF_CONSTANTS_H_ */
