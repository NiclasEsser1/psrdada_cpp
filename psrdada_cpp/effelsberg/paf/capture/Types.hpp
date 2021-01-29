#ifndef CAPTURE_TYPES_HPP_
#define CAPTURE_TYPES_HPP_

#include <vector>
#include <cstdlib>
#include <byteswap.h>
#include <iostream>     // std::cout
#include <iomanip>
#include <sstream>      // std::stringstream
#include <string>       // std::string

#include "psrdada_cpp/raw_bytes.hpp"

namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{


struct short2{
    short r;
    short i;
};


/** UDP Packet related struct **/
struct codif_t{

	static const std::size_t size = 7232;
	static const std::size_t header_size = 64;
	static const std::size_t payload_size = 7168;
	static const std::size_t samples = 128;
	static const std::size_t channels = 7;
	static const std::size_t pol = 2;
	static const std::size_t sample_size = 32;

	struct codif_hdr_t
	{
		uint32_t ref_idf : 32;
		uint32_t ref_sec : 30;
		uint32_t iscomplex : 1;
		uint32_t invalid : 1;

		uint32_t stationid : 16;
		uint32_t unassigned : 6;
		uint32_t representation : 4;
		uint32_t ref_epoch : 6;
		uint32_t samples : 24;	// Frame length (including header) divided by 8
		uint32_t bits : 5;
		uint32_t version : 3;

		uint32_t beam : 16;
		uint32_t freq : 16;
		uint32_t nchan : 16;
		uint32_t blocklength : 16;

		uint32_t reserved2 : 32;
		uint32_t period : 16;
		uint32_t reserved1 : 16;

		uint64_t totalsamples;

		uint32_t reserved3 : 32;
		uint32_t sync : 32;

		uint32_t extended2 : 32;
		uint32_t eversion : 8;
		uint32_t extended1 : 24;

		uint64_t extended_userdata : 64;
	};

	typedef codif_hdr_t HeaderType;

	codif_hdr_t hdr;

	char buffer[codif_t::size];
	short2 payload[codif_t::payload_size];

	void create(uint32_t seconds, uint32_t frame_idx, uint32_t epoch, uint32_t freq_idx, uint32_t beam_idx)
	{
		// Changing values
		hdr.ref_sec = seconds;
		hdr.ref_idf = frame_idx;
		hdr.ref_epoch = epoch;
		hdr.freq = freq_idx;
		hdr.beam = beam_idx;
		// Static values
		hdr.version = 1;
		hdr.invalid = 0;
		hdr.nchan = 7;
		hdr.samples = 896;
		hdr.iscomplex = 1;
		hdr.bits = 16;
		hdr.representation = 0;
		hdr.period = 27;
		hdr.totalsamples = 249999;
		hdr.blocklength = 6;
		hdr.sync = 0xADEADBEE;
	}

	char* serialize()
	{
		memcpy((void*)buffer, (void*)&hdr, codif_t::header_size);
		memcpy((void*)&buffer[codif_t::header_size], (void*)payload, codif_t::payload_size);
		return buffer;
	}

	void deserialize(char* buf = nullptr)
	{
		uint64_t* header;
		if(buf != nullptr){
			header = (uint64_t*)buf;
		}else{
			header = (uint64_t*)buffer;
		}
		hdr.invalid = header[0] >> 63;
		hdr.iscomplex = header[0] >> 62;
		hdr.ref_sec = header[0] >> 32;
		hdr.ref_idf = header[0] & 0x00000000FFFFFFFF;
		hdr.version = header[1] >> 61;
		hdr.bits = (header[1] & 0x1F00000000000000) >> 56;
		hdr.samples = (header[1] & 0x00FFFFFF00000000) >> 32;
		hdr.ref_epoch = (header[1] & 0x00000000FC000000) >> 26;
        hdr.representation = (header[1] & 0x0000000003C00000) >> 22;
        hdr.unassigned = (header[1] & 0x00000000003F0000) >> 16;
        hdr.stationid = header[1] & 0x000000000000FFFF;
        hdr.blocklength = header[2] >> 48;
        hdr.nchan = (header[2] & 0x0000FFFF00000000) >> 32;
        hdr.freq = (header[2] & 0x00000000FFFF0000) >> 16;
        hdr.beam = (header[2] & 0x000000000000FFFF);
        hdr.reserved1 = header[3] >> 48;
        hdr.period = (header[3] & 0x0000FFFF00000000) >> 32;
        hdr.reserved2 = (header[3] & 0x00000000FFFFFFFF);
        hdr.totalsamples = (header[4] & 0xFFFFFFFFFFFFFFFF);
        hdr.sync = header[5] >> 32;
        hdr.reserved3 = (header[5] & 0x00000000FFFFFFFF);
        hdr.extended2 = (header[6] >> 56);
        hdr.eversion = (header[6] & 0x0FFFFFFFFFFFFFFF);
        hdr.extended_userdata = (header[7] & 0xFFFFFFFFFFFFFFFF);
		// memcpy((void*)&hdr, (void*)&buf, codif_t::header_size);
		memcpy((void*)payload, (void*)&buf[codif_t::header_size], codif_t::payload_size);
	}

	void generate_payload()
	{
		for(int i = 0; i < (int)codif_t::payload_size/codif_t::sample_size; i++)
		{
			payload[i].r = rand()%255+1;
	    	payload[i].i = rand()%255+1;
		}
	}

	bool operator== (const codif_t& val)
	{
		if(hdr.ref_idf != val.hdr.ref_idf){return false;}
		if(hdr.ref_sec != val.hdr.ref_sec){return false;}
		if(hdr.iscomplex != val.hdr.iscomplex){return false;}
		if(hdr.invalid != val.hdr.invalid){return false;}
		if(hdr.stationid != val.hdr.stationid){return false;}
		if(hdr.unassigned != val.hdr.unassigned){return false;}
		if(hdr.representation != val.hdr.representation){return false;}
		if(hdr.ref_epoch != val.hdr.ref_epoch){return false;}
		if(hdr.samples != val.hdr.samples){return false;}
		if(hdr.bits != val.hdr.bits){return false;}
		if(hdr.version != val.hdr.version){return false;}
		if(hdr.beam != val.hdr.beam){return false;}
		if(hdr.freq != val.hdr.freq){return false;}
		if(hdr.nchan != val.hdr.nchan){return false;}
		if(hdr.blocklength != val.hdr.blocklength){return false;}
		if(hdr.reserved2 != val.hdr.reserved2){return false;}
		if(hdr.period != val.hdr.period){return false;}
		if(hdr.reserved1 != val.hdr.reserved1){return false;}
		if(hdr.totalsamples != val.hdr.totalsamples){return false;}
		if(hdr.reserved3 != val.hdr.reserved3){return false;}
		if(hdr.sync != val.hdr.sync){return false;}
		if(hdr.extended2 != val.hdr.extended2){return false;}
		if(hdr.eversion != val.hdr.eversion){return false;}
		if(hdr.extended1 != val.hdr.extended1){return false;}
		if(hdr.extended_userdata != val.hdr.extended_userdata){return false;}
		return true;
	}

    void print()
    {
        printf("Dataframe propertys (non-static values)\n");
        printf("Dataframe index: %u\n", hdr.ref_idf);
        printf("Seconds from ref epoch: %u\n", hdr.ref_sec);
        printf("Ref epoch: %u\n", hdr.ref_epoch);
        printf("Frequency: %u\n", hdr.freq);
        printf("Beam index: %u\n", hdr.beam);
    }
};








template<typename Protocol>
struct DataFrame{
    // RawBytes& block;
    // bool operator()(RawBytes &block){};

    Protocol packet;
	typedef typename Protocol::HeaderType HeaderType;

	DataFrame(){

	}

	void set_header(HeaderType& header)
	{
		packet.hdr = header;
	}


	void set_payload(char* pay)
	{
		packet.payload = pay;
	}

  	void deserialize(char* buf)
	{
		packet.deserialize(buf);
  	}

  	char* serialize()
	{
      	return packet.serialize();
  	}

	HeaderType hdr()
	{
		return packet.hdr;
	}

};

/** Confguration **/
// template <typename PacketType>
struct capture_conf_t{
    /** Dada key **/
    key_t key;

    /** Address where to listen **/
    std::string capture_addr;

    std::string capture_ctrl_addr;

    std::string psrdada_header_file;

    std::string log;

    std::size_t capture_ctrl_port;

    std::size_t capture_ctrl_cpu_bind;

    std::size_t buffer_ctrl_cpu_bind;

    std::size_t dataframe_ref;

    std::size_t sec_ref;

    std::size_t ref_epoch;

    std::size_t nbeam;

    std::vector<std::size_t> capture_ports;

    std::vector<std::size_t> capture_cpu_bind;

    std::size_t frame_size;

    std::size_t offset;

    std::size_t n_buffers;

    std::size_t rbuffer_hdr_size;

    std::size_t rbuffer_size;

    std::size_t tbuffer_size;

    std::size_t nframes_tmp_buffer;

    std::size_t threshold;  // Number of frames to wait for old frames

    std::size_t n_catcher;  // number of catching threads

    codif_t hdr_ref;

    void print()
    {
        std::cout << "\n------------------------\n"
            << "CAPTURE CONFIG:"
            << "\n------------------------\n" << std::endl;
        // PSRDADA ringbuffer key
        std::cout << "Ringbuffer key: " << std::hex << key << std::endl;
        // Catcher
        for(int i = 0; i < capture_ports.size(); i++)
        {
            std::cout << "Catcher " << std::to_string(i) << "\ton  address: " << capture_addr << ":" << std::to_string(capture_ports.at(i)) << std::flush;
            if(capture_cpu_bind.at(i) != -1){
                std::cout << "\tbind to CPU " << std::to_string(capture_cpu_bind.at(i)) << std::endl;
            }else{
                std::cout << "\tnot binded to CPU " << std::endl;
            }
        }
        // Capture controller
        std::cout << "Capture ctrl\ton  address: " << capture_ctrl_addr << ":" << std::to_string(capture_ctrl_port) << std::flush;
        if(capture_ctrl_cpu_bind != -1){
            std::cout << "\tbind to CPU " << std::to_string(capture_ctrl_cpu_bind) << std::endl;
        }else{
            std::cout << "\tnot binded to CPU " << std::endl;
        }
        //Buffer controller
        std::cout << "Buffer ctrl\t\t\t\t\t" << std::flush;
        if(buffer_ctrl_cpu_bind != -1){
            std::cout << "bind to CPU " << std::to_string(capture_ctrl_cpu_bind) << std::endl;
        }else{
            std::cout << "not binded to CPU " << std::endl;
        }
        // Beams
        std::cout << "Going to capture\t" << std::to_string(nbeam) << " dual-polarized  elements" << std::endl;
        // Reference
        if(dataframe_ref != -1 && sec_ref != -1){
            std::cout << "Reference frame " << std::to_string(dataframe_ref) << std::endl;
            std::cout << "Reference second " << std::to_string(sec_ref) << std::endl;
        }else{
            std::cout << "No reference provided, will be determined automatically" << std::endl;
        }
        std::cout << "Header file in " << psrdada_header_file << std::endl;
    }
};


}
}
}
}

#endif

// void deserialize(char* in)
// {
//     uint64_t *p, writebuf;
//     p = (uint64_t*)in;
//     writebuf = bswap_64(*p);
//     ref_idf = (std::size_t)writebuf & 0x00000000ffffffff;
//     hdr.ref_sec = (std::size_t)(writebuf & 0x3fffffff00000000) >> 32;
//     // valid = (std::size_t)(writebuf & 0x8000000000000000) >> 63;
//     writebuf = bswap_64(*(p + 1));
//     ref_epoch = (std::size_t)(writebuf & 0x00000000fc000000) >> 26;
//     writebuf = bswap_64(*(p + 2));
//     hdr.freq = (float)((writebuf & 0x00000000ffff0000) >> 16);
//     hdr.beam = writebuf & 0x000000000000ffff;
// // }
//
//
// struct codif_hdr_t{
//     // Word 1
//     /** 0 the data frame is not valied, 1 the data frame is valied; **/
//     // bool valid = 0;
//     //
//     // bool complex = 0;
//     /** Secs from reference epoch at start of period; **/
//     std::size_t ref_sec = 0;
//     /** data frame number in one period; **/
//     std::size_t ref_idf = 0;
//
//     // Word 2
//     /** CODIF version **/
//     // unsigned char version = 0;
//     /** Sample bit size **/
//     // int bit_sz = 0;
//     /** Length of array **/
//     // int array_length = 0;
//     /** Number of half a year from 1st of January, 2000 for the reference epochch; **/
//     std::size_t ref_epoch = 0;
//     /** reprensentaiton of payload data **/
//     // int reprensent = 0;
//     /** Station id of telescope **/
//     // unsigned char station_id[2]
//     /** The id of beam, counting from 0; **/
//     int beam = 0;
//     /** Frequency of the first chunnal in each block (integer MHz); **/
//     float freq = 0;
//
//     uint64_t words[8];
//
//     void deserialize(std::stringstream& is)
//     {
//         is >> (uint64_t) word[0];
//         is >> (uint64_t) word[1];
//         is >> (uint64_t) word[2];
//         is >> (uint64_t) word[3];
//         is >> (uint64_t) word[4];
//         is >> (uint64_t) word[5];
//         is >> (uint64_t) word[6];
//         is >> (uint64_t) word[7];
//
//         ref_idf = (std::size_t)bswap_64(word[0]) & 0x00000000ffffffff;
//         ref_sec = (std::size_t)(bswap_64(word[0]) & 0x3fffffff00000000) >> 32;
//         // valid = (std::size_t)(writebuf & 0x8000000000000000) >> 63;
//
//         ref_epoch = (std::size_t)(bswap_64(word[1]) & 0x00000000fc000000) >> 26;
//
//         freq = (float)((bswap_64(word[2]) & 0x00000000ffff0000) >> 16);
//         beam = bswap_64(word[2]) & 0x000000000000ffff;
//     }
//
//     std::stringstream serialize()
//     {
//         std::stringstream os;
//         os << (uint64_t)((hdr.ref_idf & 0x00000000ffffffff)
//             & ((hdr.ref_sec & 0x3fffffff00000000) << 32));
//         os << (uint64_t)(((hdr.ref_epoch & 0x00000000fc000000) << 26) );
//         os << (uint64_t)((hdr.beam & 0x000000000000ffff)
//             & ((hdr.freq & 0x00000000ffff0000) << 16));
//         os << (uint64_t)(0x0000000000000000);
//         os << (uint64_t)(0x0000000000000000);
//         os << (uint64_t)(0x0000000000000000);
//         os << (uint64_t)(0x0000000000000000);
//         os << (uint64_t)(0x0000000000000000);
//         return os;
//     }
//
// };
