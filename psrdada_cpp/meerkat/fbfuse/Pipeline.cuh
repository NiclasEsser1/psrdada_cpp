#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_PIPELINE_HPP
#define PSRDADA_CPP_MEERKAT_FBFUSE_PIPELINE_HPP

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {

template <typename Handler>
class Pipeline
{
private:
    typedef long double TimeType;

public:
    Pipeline(PipelineConfig const& config,
        DadaWriteClient& cb_writer,
        DadaWriteClient& ib_writer,
        std::size_t input_data_buffer_size);
    ~Pipeline();
    Pipeline(Pipeline const&) = delete;

    void init(RawBlock& header);
    bool operator()(RawBlock& data);

private:
    void set_header(RawBlock& header);

private:
    PipelineConfig const& _config;

    std:size_t _sample_clock_start;
    long double _sample_clock;
    long double _sync_time;
    long double _unix_timestamp;
    std::size_t _sample_clock_tick_per_block;
    std::size_t _call_count;

    DoubleDeviceBuffer<char2> _taftp_db; // Input from F-engine
    DoubleDeviceBuffer<char> _tbftf_db; // Output of coherent beamformer
    DoubleDeviceBuffer<char> _tftf_db; // Output of incoherent beamformer

    DadaWriteClient& _cb_writer;
    DadaWriteClient::HeaderStream& _cb_header_stream;
    DadaWriteClient::DataStream& _cb_header_stream;
    DadaWriteClient& _ib_writer;
    DadaWriteClient::HeaderStream& _ib_header_stream;
    DadaWriteClient::DataStream& _ib_header_stream;

    cudaStream_t _h2d_copy_stream;
    cudaStream_t _processing_stream;
    cudaStream_t _d2h_copy_stream;

    std::size_t _nheap_groups_per_block;
    std::size_t _nsamples_per_dada_block;
    std::unique_ptr<DelayManager> delay_manager;
    std::unique_ptr<WeightsManager> weights_manager;

};

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MEERKAT_FBFUSE_PIPELINE_HPP