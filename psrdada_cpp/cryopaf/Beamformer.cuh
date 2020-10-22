#ifndef BEAMFORMER_H_
#define BEAMFORMER_H_

#include <cuda.h>

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{



class Beamformer
{
// Methods
public:
	Beamformer(bf_config_t *conf, int device_id)
		: _conf(conf), _device_id(device_id){}

	virtual void init(bf_config_t *conf = nullptr) = 0;

	void process();
	void upload_weights();

	int device_id() { return _device_id; }
	bool is_init() { return _success; }

	std::string device_name() { return _name; }
	std::size_t shared_mem_size() { return _shared_mem_total; }

	cudaStream_t h2d_stream() { return _h2d_stream; }
	cudaStream_t proc_stream() { return _proc_stream; }
	cudaStream_t d2h_stream() { return _d2h_stream; }
	cudaDeviceProp device_prop() { return _prop; }

	dim3 grid() { return _grid_layout; }
	dim3 block() {return _block_layout; }


	void print_layout()
	{
		std::cout << " Kernel layout: " << std::endl
			<< " g.x = " << std::to_string(_grid_layout.x) << std::endl
			<< " g.y = " << std::to_string(_grid_layout.y) << std::endl
			<< " g.z = " << std::to_string(_grid_layout.z) << std::endl
			<< " b.x = " << std::to_string(_block_layout.x)<< std::endl
			<< " b.y = " << std::to_string(_block_layout.y)<< std::endl
			<< " b.z = " << std::to_string(_block_layout.z)<< std::endl;
	}
protected:
	void check_shared_mem_size()
	{
		std::cout << "Required shared memory: " << std::to_string(_shared_mem_total) << " Bytes" << std::endl;
		if(_prop.sharedMemPerBlock < _shared_mem_total)
		{
			std::cout << "The requested size for shared memory per block exceeds the size provided by device "
				<< std::to_string(_device_id) << std::endl
				<< "Warning: Kernel will not get launched !" << std::endl;
				_success = false;
		}else{
			_success = true;
		}
	}



// Attributes
protected:

	int _device_id;
	bool _success = true;
	bf_config_t *_conf;	// TODO: non pointer


	cudaDeviceProp _prop;
	std::string _name;
	std::size_t _shared_mem_static;
	std::size_t _shared_mem_dynamic;
	std::size_t _shared_mem_total;

	dim3 _grid_layout;
	dim3 _block_layout;

	cudaStream_t _h2d_stream;
	cudaStream_t _proc_stream;
	cudaStream_t _d2h_stream;
};


}
}
}


#endif
