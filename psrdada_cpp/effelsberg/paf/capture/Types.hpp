#ifndef CAPTURE_TYPES_HPP_
#define CAPTURE_TYPES_HPP_

#include <vector>
#include <cstdlib>
#include <byteswap.h>

#include "psrdada_cpp/raw_bytes.hpp"

namespace psrdada_cpp {
namespace effelsberg{
namespace paf{
namespace capture{

/** UDP Packet related struct **/
// Currently just support for CODIF
struct codif_t{

    static const std::size_t size = 7234;
    static const std::size_t header_size = 64;
    static const std::size_t payload_size = 7168;
    static const std::size_t reserved = 2;

    char* ptr;
    /** 0 the data frame is not valied, 1 the data frame is valied; **/
    bool valid;

    bool complex;

    /** Sample bit size **/
    int bit_sz;

    /** data frame number in one period; **/
    std::size_t idf;

    /** Secs from reference epoch at start of period; **/
    std::size_t sec;

    /** Number of half a year from 1st of January, 2000 for the reference epochch; **/
    std::size_t epoch_ref;


    /** The id of beam, counting from 0; **/
    int beam;

    /** Frequency of the first chunnal in each block (integer MHz); **/
    double freq;

    int parse(char* bytes)
    {
        uint64_t *p, writebuf;
        p = (uint64_t*)bytes;

        writebuf = bswap_64(*p);
        idf = (std::size_t)writebuf & 0x00000000ffffffff;
        sec = (std::size_t)(writebuf & 0x3fffffff00000000) >> 32;
        valid = (std::size_t)(writebuf & 0x8000000000000000) >> 63;

        writebuf = bswap_64(*(p + 1));
        epoch_ref = (std::size_t)(writebuf & 0x00000000fc000000) >> 26;

        writebuf = bswap_64(*(p + 2));
        freq = (double)((writebuf & 0x00000000ffff0000) >> 16);
        beam = writebuf & 0x000000000000ffff;
    }

    void init()
    {
      valid = 0;
      idf = 0;
      sec = 0;
      epoch_ref = 0;
      beam = 0;
      freq = .0;
    }
};



template<typename Protocol>
struct DataFrame{
    // RawBytes& block;
    // bool operator()(RawBytes &block){};

    char* payload;
    std::vector<char> packet;

    Protocol hdr;

    DataFrame(){
        packet.resize(Protocol::size);
        hdr.init();
        hdr.ptr = &packet[0];
        payload = &packet[Protocol::header_size];
    }

    void update(){
        hdr.parse(packet.data());
    }

    void generate(char* header_bytes){
        hdr.parse(header_bytes);
        for(std::size_t ii = 0; ii < Protocol::payload_size; ++ii)
    	{
    	    payload[ii] = rand()%255+1;
    	}
    }

    Protocol* header(){return &hdr;}
};

/** psrdada rinbuffer related struct **/
// struct dada_hdr_t{
//     std::size_t num_buffer;
//
//     std::size_t rbuffer_size;
//
//     PsrdadaHeader(/** DADA Client **/)
//     {
//
//     }
// };

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

    std::size_t epoch_ref;

    std::size_t nbeam;

    std::vector<std::size_t> capture_ports;

    std::vector<std::size_t> capture_cpu_bind;


    std::size_t frame_size;

    std::size_t offset;

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
