#ifndef CAPTURE_TYPES_HPP_
#define CAPTURE_TYPES_HPP_

#include <vector>
#include <cstdlib>
#include <byteswap.h>
#include <iostream>     // std::cout
#include <iomanip>
#include <sstream>      // std::stringstream
#include <string>       // std::string

#include <boost/date_time/posix_time/posix_time.hpp> //include all types plus i/o
#include <boost/date_time.hpp>

#include "psrdada_cpp/raw_bytes.hpp"

namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{


using namespace boost::posix_time;
using namespace boost::gregorian;

static const std::map<int, ptime> epochs = {
	{51, ptime(date(2025, 07, 01), time_duration(00,00,00))},
	{50, ptime(date(2025, 01, 01), time_duration(00,00,00))},
	{49, ptime(date(2024, 07, 01), time_duration(00,00,00))},
	{48, ptime(date(2024, 01, 01), time_duration(00,00,00))},
	{47, ptime(date(2023, 07, 01), time_duration(00,00,00))},
	{46, ptime(date(2023, 01, 01), time_duration(00,00,00))},
	{45, ptime(date(2022, 07, 01), time_duration(00,00,00))},
	{44, ptime(date(2022, 01, 01), time_duration(00,00,00))},
	{43, ptime(date(2021, 07, 01), time_duration(00,00,00))},
	{42, ptime(date(2021, 01, 01), time_duration(00,00,00))},
	{41, ptime(date(2020, 07, 01), time_duration(00,00,00))},
	{40, ptime(date(2020, 01, 01), time_duration(00,00,00))},
	{39, ptime(date(2019, 07, 01), time_duration(00,00,00))},
	{38, ptime(date(2019, 01, 01), time_duration(00,00,00))},
	{37, ptime(date(2018, 07, 01), time_duration(00,00,00))},
	{36, ptime(date(2018, 01, 01), time_duration(00,00,00))}
};

struct short2{
    short r;
    short i;

	template<typename archive>
	void serialize(archive& ar, const unsigned)
	{
		ar << (short)r;
		ar << (short)i;
	}
};


/** UDP Packet related struct **/
struct codif_t{

	static const std::size_t size = 7232;
	static const std::size_t header_size = 64;
	static const std::size_t payload_size = 7168;
	static const std::size_t time = 128;
	static const std::size_t fs = 1185185;
	static const std::size_t channels = 7;
	static const std::size_t pol = 2;
	static const std::size_t samples = codif_t::time * codif_t::channels * codif_t::pol;
	static const std::size_t sample_size = 32;
	static const std::size_t frame_resolution = (codif_t::fs/codif_t::time);

	struct codif_hdr_t
	{
		uint32_t ref_idf : 32;
		uint32_t sec_from_epoch : 30;
		uint32_t iscomplex : 1;
		uint32_t invalid : 1;

		uint32_t version : 3;
		uint32_t bits : 5;
		uint32_t length : 24;	// Frame length (including header) divided by 8
		uint32_t ref_epoch : 6;
		uint32_t representation : 4;
		uint32_t unassigned : 6;
		uint32_t stationid : 16;

		uint32_t blocklength : 16;
		uint32_t nchan : 16;
		uint32_t freq : 16;
		uint32_t beam : 16;

		uint32_t reserved1 : 16;
		uint32_t period : 16;
		uint32_t reserved2 : 32;

		uint64_t totalsamples;

		uint32_t sync : 32;
		uint32_t reserved3 : 32;

		uint32_t eversion : 8;
		uint32_t extended1 : 24;
		uint32_t extended2 : 32;

		uint64_t extended_userdata : 64;
	};
	// Default init of header
	codif_hdr_t hdr = {0,1,0,0, 1,16,896,0,0,0,20555, 6,13,0,0, 0,27,0, 249999, 0xadeadbee,0, 0,0,0, 0};

	typedef codif_hdr_t HeaderType;
	short2 payload[codif_t::samples];
	unsigned char buffer[codif_t::size];

	void serialize()
	{
		uint64_t header[8];
		header[0] = bswap_64((uint64_t)hdr.invalid << 63
			| (uint64_t)hdr.iscomplex << 62
			| (uint64_t)hdr.sec_from_epoch << 32
			| (uint64_t)hdr.ref_idf);
		header[1] = bswap_64((uint64_t)hdr.version << 61
			| (uint64_t)hdr.bits << 56
			| (uint64_t)hdr.length << 32
			| (uint64_t)hdr.ref_epoch << 26
			| (uint64_t)hdr.representation << 22
			| (uint64_t)hdr.unassigned << 16
			| (uint64_t)hdr.stationid);
		header[2] = bswap_64((uint64_t)hdr.blocklength << 48
			| (uint64_t)hdr.nchan << 32
			| (uint64_t)hdr.freq << 16
			| (uint64_t)hdr.beam);
		header[3] = bswap_64((uint64_t)hdr.reserved1 << 48
			| (uint64_t)hdr.period << 32
			| (uint64_t)hdr.reserved2);
		header[4] = bswap_64((uint64_t)hdr.totalsamples);
		header[5] = bswap_64((uint64_t)hdr.sync << 32
			| (uint64_t)hdr.reserved3);
		header[6] = bswap_64((uint64_t)hdr.eversion << 56
			| (uint64_t)hdr.extended1 << 32
			| (uint64_t)hdr.extended2);
		header[7] = ((uint64_t)hdr.extended_userdata);
		memcpy(buffer, (void*)&header, sizeof(codif_hdr_t));
		memcpy(&buffer[sizeof(codif_hdr_t)], (void*)payload, codif_t::payload_size);
	}

	void deserialize()
	{
		uint64_t* header = (uint64_t*)buffer;
		hdr.invalid = bswap_64(header[0]) >> 63;
		hdr.iscomplex = bswap_64(header[0]) >> 62;
		hdr.sec_from_epoch = bswap_64(header[0]) >> 32;
		hdr.ref_idf = bswap_64(header[0]) & 0x00000000FFFFFFFF;
		hdr.version = bswap_64(header[1]) >> 61;
		hdr.bits = (bswap_64(header[1]) & 0x1F00000000000000) >> 56;
		hdr.length = (bswap_64(header[1]) & 0x00FFFFFF00000000) >> 32;
		hdr.ref_epoch = (bswap_64(header[1]) & 0x00000000FC000000) >> 26;
        hdr.representation = (bswap_64(header[1]) & 0x0000000003C00000) >> 22;
        hdr.unassigned = (bswap_64(header[1]) & 0x00000000003F0000) >> 16;
        hdr.stationid = bswap_64(header[1]) & 0x000000000000FFFF;
        hdr.blocklength = bswap_64(header[2]) >> 48;
        hdr.nchan = (bswap_64(header[2]) & 0x0000FFFF00000000) >> 32;
        hdr.freq = (bswap_64(header[2]) & 0x00000000FFFF0000) >> 16;
        hdr.beam = (bswap_64(header[2]) & 0x000000000000FFFF);
        hdr.reserved1 = bswap_64(header[3]) >> 48;
        hdr.period = (bswap_64(header[3]) & 0x0000FFFF00000000) >> 32;
        hdr.reserved2 = (bswap_64(header[3]) & 0x00000000FFFFFFFF);
        hdr.totalsamples = (bswap_64(header[4]) & 0xFFFFFFFFFFFFFFFF);
        hdr.sync = bswap_64(header[5]) >> 32;
        hdr.reserved3 = (bswap_64(header[5]) & 0x00000000FFFFFFFF);
        hdr.extended2 = (bswap_64(header[6]) >> 56);
        hdr.eversion = (bswap_64(header[6]) & 0x0FFFFFFFFFFFFFFF);
        hdr.extended_userdata = (bswap_64(header[7]) & 0xFFFFFFFFFFFFFFFF);
		memcpy((void*)payload, (void*)&buffer[codif_t::header_size], codif_t::payload_size);
	}

	void sync_frame()
	{
		uint32_t frames_since_epoch;
		uint32_t seconds_since_epoch = 15768000; // Maximum is an half year in seconds
		uint32_t epoch_ref;

		time_duration dur;
		ptime now = second_clock::universal_time();

		for (std::pair<int, ptime> epoch : epochs)
		{
			dur = now - epoch.second;
			if(epoch.second < now && seconds_since_epoch > dur.total_seconds())
			{
				seconds_since_epoch = (uint32_t)dur.total_seconds();
				hdr.ref_epoch = epoch.first;
			}
		}
		hdr.sec_from_epoch = floor(seconds_since_epoch / hdr.period) * hdr.period;
		hdr.ref_idf = (int)(floor(seconds_since_epoch - hdr.sec_from_epoch) * codif_t::fs / codif_t::time);
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
		if(hdr.sec_from_epoch != val.hdr.sec_from_epoch){return false;}
		if(hdr.iscomplex != val.hdr.iscomplex){return false;}
		if(hdr.invalid != val.hdr.invalid){return false;}
		if(hdr.stationid != val.hdr.stationid){return false;}
		if(hdr.unassigned != val.hdr.unassigned){return false;}
		if(hdr.representation != val.hdr.representation){return false;}
		if(hdr.ref_epoch != val.hdr.ref_epoch){return false;}
		if(hdr.length != val.hdr.length){return false;}
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
        printf("Dataframe propertys\n");
        printf("Dataframe index: %u\n", hdr.ref_idf);
        printf("Seconds from ref sec_from_epoch: %u\n", hdr.sec_from_epoch);
        printf("Ref sec_from_epoch: %u\n", hdr.ref_epoch);
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

  	void deserialize()
	{
		return packet.deserialize();
  	}

  	void serialize()
	{
      	packet.serialize();
  	}

	unsigned char* buffer()
	{
		return packet.buffer;
	}
	HeaderType hdr()
	{
		return packet.hdr;
	}

	void print()
	{
		packet.print();
	}

	void sync_frame()
	{
		packet.sync_frame();
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
