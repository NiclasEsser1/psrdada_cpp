#ifndef BITMAP_HPP
#define BITMAP_HPP

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdint.h>

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{

class Bitmap_IO {
public:
	Bitmap_IO( int w, int h, bool allocate = true)
	{
		if(allocate)
			image = new char[ w * h * 3];
		header.filesz = sizeof( bmpHeader ) + sizeof( bmpInfo ) + ( w * h * 3);
		header.bmp_offset = sizeof( bmpHeader ) + sizeof( bmpInfo );
		info.header_sz = sizeof( bmpInfo );
		info.width = w;
		info.height = h;
		info.nplanes = 1;
		info.bitspp = 24;
		info.compress_type = 0;
		info.bmp_bytesz = w * h * 3;
		info.hres = 2835;
		info.vres = 2835;
		info.ncolors = 1;
		info.nimpcolors = 0;
	}

	bool save(std::string filename )
	{
		char head[2] = {'B', 'M'};
		unsigned char bmppad[3] = {0,0,0};
		if( image == nullptr )
		{
			std::cerr << "Image unitialized" << std::endl;
			return false;
		}
		// printf("DIR: %s\n", filename);

		FILE* fid = fopen(filename.c_str(), "wb");
		if(fid != NULL)
		{
			fwrite(head, sizeof(char), 2, fid);
			fwrite(&header, sizeof( bmpHeader ), 1, fid);
			fwrite(&info, sizeof( bmpInfo ), 1, fid);
			for(int i = height()-1; i >= 0; i--)
			{
				fwrite(&bmppad, sizeof(char), (4-(width()*3)%4)%4, fid);
				fwrite(&image[width()*i*3], sizeof(char)* colorDepth()/8, width(), fid);
			}
			fclose(fid);
		}
		else
		{
			std::cout << "Could not open file: " << filename << std::endl;
		}

		return true;
	}

	bool load(const char* filename)
	{
		if( image != NULL )
		{
			delete[] image;
		}

		std::ifstream file( filename, std::ios::in | std::ios::binary );

		if( !file.is_open() )
		{
			std::cerr << "Cannot open " << filename << std::endl;
			return false;
		}

		char BM[ 2 ];
		file.read( (char*)( BM ), 2 );

		file.read( (char*)( &header ), sizeof( bmpHeader ) );

		file.read( (char*)( &info ), sizeof( bmpInfo ) );

		file.seekg( header.bmp_offset, std::ios::beg );

		image = new char[ info.width * info.height ];

		file.read(image, info.width * info.height);

		file.close();

		return true;
	}

	~Bitmap_IO()
	{
	}

	int width() {
		return info.width;
	}

	int height() {
		return info.height;
	}

	int colorDepth() {
		return info.bitspp;
	}

	char getPixel( int x, int y ) {
		return image[ y * info.width + x ];
	}

	void setPixel( int x, int y, char color ) {
		image[ y * info.width + x ] = color;
	}
	char* getImagePtr(int pos = 0) {
		return &image[pos];
	}
	void setImagePtr(char* input){
		image = input;
	}



private:
	struct  bmpHeader {
		uint32_t filesz;
		uint16_t creator1;
		uint16_t creator2;
		uint32_t bmp_offset;
	};

	struct bmpInfo {
		uint32_t header_sz;
		int32_t width;
		int32_t height;
		uint16_t nplanes;
		uint16_t bitspp;
		uint32_t compress_type;
		uint32_t bmp_bytesz;
		int32_t hres;
		int32_t vres;
		uint32_t ncolors;
		uint32_t nimpcolors;
	};
	char* image;
	bmpHeader header;
	bmpInfo info;

};

}
}
}

#endif
