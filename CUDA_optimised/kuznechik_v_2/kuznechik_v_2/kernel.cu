//Массив long long, копируем, увеличиваем, потом перепаковать
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include "tests_gost.h"
#include <stdio.h>


#include <iostream>
using namespace std;

#define ECB_CODE 1
#define CTR_CODE 2
#define OFB_CODE 3
#define CBC_CODE 4
#define CFB_CODE 5

__constant__ unsigned char pi_table[256] = {
	0xFC, 0xEE, 0xDD, 0x11, 0xCF, 0x6E, 0x31, 0x16,
	 0xFB, 0xC4, 0xFA, 0xDA, 0x23, 0xC5, 0x04, 0x4D,
	0xE9, 0x77, 0xF0, 0xDB, 0x93, 0x2E, 0x99, 0xBA,
	 0x17, 0x36, 0xF1, 0xBB, 0x14, 0xCD, 0x5F, 0xC1,
	0xF9, 0x18, 0x65, 0x5A, 0xE2, 0x5C, 0xEF, 0x21,
	 0x81, 0x1C, 0x3C, 0x42, 0x8B, 0x01, 0x8E, 0x4F,
	0x05, 0x84, 0x02, 0xAE, 0xE3, 0x6A, 0x8F, 0xA0,
	 0x06, 0x0B, 0xED, 0x98, 0x7F, 0xD4, 0xD3, 0x1F,
	0xEB, 0x34, 0x2C, 0x51, 0xEA, 0xC8, 0x48, 0xAB,
	 0xF2, 0x2A, 0x68, 0xA2, 0xFD, 0x3A, 0xCE, 0xCC,
	0xB5, 0x70, 0x0E, 0x56, 0x08, 0x0C, 0x76, 0x12,
	 0xBF, 0x72, 0x13, 0x47, 0x9C, 0xB7, 0x5D, 0x87,
	0x15, 0xA1, 0x96, 0x29, 0x10, 0x7B, 0x9A, 0xC7,
	 0xF3, 0x91, 0x78, 0x6F, 0x9D, 0x9E, 0xB2, 0xB1,
	0x32, 0x75, 0x19, 0x3D, 0xFF, 0x35, 0x8A, 0x7E,
	 0x6D, 0x54, 0xC6, 0x80, 0xC3, 0xBD, 0x0D, 0x57,
	0xDF, 0xF5, 0x24, 0xA9, 0x3E, 0xA8, 0x43, 0xC9,
	 0xD7, 0x79, 0xD6, 0xF6, 0x7C, 0x22, 0xB9, 0x03,
	0xE0, 0x0F, 0xEC, 0xDE, 0x7A, 0x94, 0xB0, 0xBC,
	 0xDC, 0xE8, 0x28, 0x50, 0x4E, 0x33, 0x0A, 0x4A,
	0xA7, 0x97, 0x60, 0x73, 0x1E, 0x00, 0x62, 0x44,
	 0x1A, 0xB8, 0x38, 0x82, 0x64, 0x9F, 0x26, 0x41,
	0xAD, 0x45, 0x46, 0x92, 0x27, 0x5E, 0x55, 0x2F,
	 0x8C, 0xA3, 0xA5, 0x7D, 0x69, 0xD5, 0x95, 0x3B,
	0x07, 0x58, 0xB3, 0x40, 0x86, 0xAC, 0x1D, 0xF7,
	 0x30, 0x37, 0x6B, 0xE4, 0x88, 0xD9, 0xE7, 0x89,
	0xE1, 0x1B, 0x83, 0x49, 0x4C, 0x3F, 0xF8, 0xFE,
	 0x8D, 0x53, 0xAA, 0x90, 0xCA, 0xD8, 0x85, 0x61,
	0x20, 0x71, 0x67, 0xA4, 0x2D, 0x2B, 0x09, 0x5B,
	 0xCB, 0x9B, 0x25, 0xD0, 0xBE, 0xE5, 0x6C, 0x52,
	0x59, 0xA6, 0x74, 0xD2, 0xE6, 0xF4, 0xB4, 0xC0,
	 0xD1, 0x66, 0xAF, 0xC2, 0x39, 0x4B, 0x63, 0xB6
};
__constant__ unsigned char reverse_pi_table[256] = {
0xA5, 0x2D, 0x32, 0x8F, 0x0E, 0x30, 0x38, 0xC0,
 0x54, 0xE6, 0x9E, 0x39, 0x55, 0x7E, 0x52, 0x91,
0x64, 0x03, 0x57, 0x5A, 0x1C, 0x60, 0x07, 0x18,
 0x21, 0x72, 0xA8, 0xD1, 0x29, 0xC6, 0xA4, 0x3F,
0xE0, 0x27, 0x8D, 0x0C, 0x82, 0xEA, 0xAE, 0xB4,
 0x9A, 0x63, 0x49, 0xE5, 0x42, 0xE4, 0x15, 0xB7,
0xC8, 0x06, 0x70, 0x9D, 0x41, 0x75, 0x19, 0xC9,
 0xAA, 0xFC, 0x4D, 0xBF, 0x2A, 0x73, 0x84, 0xD5,
0xC3, 0xAF, 0x2B, 0x86, 0xA7, 0xB1, 0xB2, 0x5B,
 0x46, 0xD3, 0x9F, 0xFD, 0xD4, 0x0F, 0x9C, 0x2F,
0x9B, 0x43, 0xEF, 0xD9, 0x79, 0xB6, 0x53, 0x7F,
 0xC1, 0xF0, 0x23, 0xE7, 0x25, 0x5E, 0xB5, 0x1E,
0xA2, 0xDF, 0xA6, 0xFE, 0xAC, 0x22, 0xF9, 0xE2,
 0x4A, 0xBC, 0x35, 0xCA, 0xEE, 0x78, 0x05, 0x6B,
0x51, 0xE1, 0x59, 0xA3, 0xF2, 0x71, 0x56, 0x11,
 0x6A, 0x89, 0x94, 0x65, 0x8C, 0xBB, 0x77, 0x3C,
0x7B, 0x28, 0xAB, 0xD2, 0x31, 0xDE, 0xC4, 0x5F,
 0xCC, 0xCF, 0x76, 0x2C, 0xB8, 0xD8, 0x2E, 0x36,
0xDB, 0x69, 0xB3, 0x14, 0x95, 0xBE, 0x62, 0xA1,
 0x3B, 0x16, 0x66, 0xE9, 0x5C, 0x6C, 0x6D, 0xAD,
0x37, 0x61, 0x4B, 0xB9, 0xE3, 0xBA, 0xF1, 0xA0,
 0x85, 0x83, 0xDA, 0x47, 0xC5, 0xB0, 0x33, 0xFA,
0x96, 0x6F, 0x6E, 0xC2, 0xF6, 0x50, 0xFF, 0x5D,
0xA9, 0x8E, 0x17, 0x1B, 0x97, 0x7D, 0xEC, 0x58,
0xF7, 0x1F, 0xFB, 0x7C, 0x09, 0x0D, 0x7A, 0x67,
0x45, 0x87, 0xDC, 0xE8, 0x4F, 0x1D, 0x4E, 0x04,
0xEB, 0xF8, 0xF3, 0x3E, 0x3D, 0xBD, 0x8A, 0x88,
0xDD, 0xCD, 0x0B, 0x13, 0x98, 0x02, 0x93, 0x80,
0x90, 0xD0, 0x24, 0x34, 0xCB, 0xED, 0xF4, 0xCE,
0x99, 0x10, 0x44, 0x40, 0x92, 0x3A, 0x01, 0x26,
0x12, 0x1A, 0x48, 0x68, 0xF5, 0x81, 0x8B, 0xC7,
 0xD6, 0x20, 0x0A, 0x08, 0x00, 0x4C, 0xD7, 0x74
};


__constant__ unsigned char key_initial_consts[512] = {
0x6e, 0xa2, 0x76, 0x72, 0x6c, 0x48, 0x7a, 0xb8, 0x5d, 0x27, 0xbd, 0x10, 0xdd, 0x84, 0x94, 0x01,
0xdc, 0x87, 0xec, 0xe4, 0xd8, 0x90, 0xf4, 0xb3, 0xba, 0x4e, 0xb9, 0x20, 0x79, 0xcb, 0xeb, 0x02,
0xb2, 0x25, 0x9a, 0x96, 0xb4, 0xd8, 0x8e, 0x0b, 0xe7, 0x69, 0x04, 0x30, 0xa4, 0x4f, 0x7f, 0x03,
0x7b, 0xcd, 0x1b, 0x0b, 0x73, 0xe3, 0x2b, 0xa5, 0xb7, 0x9c, 0xb1, 0x40, 0xf2, 0x55, 0x15, 0x04,
0x15, 0x6f, 0x6d, 0x79, 0x1f, 0xab, 0x51, 0x1d, 0xea, 0xbb, 0x0c, 0x50, 0x2f, 0xd1, 0x81, 0x05,
0xa7, 0x4a, 0xf7, 0xef, 0xab, 0x73, 0xdf, 0x16, 0x0d, 0xd2, 0x08, 0x60, 0x8b, 0x9e, 0xfe, 0x06,
0xc9, 0xe8, 0x81, 0x9d, 0xc7, 0x3b, 0xa5, 0xae, 0x50, 0xf5, 0xb5, 0x70, 0x56, 0x1a, 0x6a, 0x07,
0xf6, 0x59, 0x36, 0x16, 0xe6, 0x05, 0x56, 0x89, 0xad, 0xfb, 0xa1, 0x80, 0x27, 0xaa, 0x2a, 0x08,
0x98, 0xfb, 0x40, 0x64, 0x8a, 0x4d, 0x2c, 0x31, 0xf0, 0xdc, 0x1c, 0x90, 0xfa, 0x2e, 0xbe, 0x09,
0x2a, 0xde, 0xda, 0xf2, 0x3e, 0x95, 0xa2, 0x3a, 0x17, 0xb5, 0x18, 0xa0, 0x5e, 0x61, 0xc1, 0x0a,
0x44, 0x7c, 0xac, 0x80, 0x52, 0xdd, 0xd8, 0x82, 0x4a, 0x92, 0xa5, 0xb0, 0x83, 0xe5, 0x55, 0x0b,
0x8d, 0x94, 0x2d, 0x1d, 0x95, 0xe6, 0x7d, 0x2c, 0x1a, 0x67, 0x10, 0xc0, 0xd5, 0xff, 0x3f, 0x0c,
0xe3, 0x36, 0x5b, 0x6f, 0xf9, 0xae, 0x07, 0x94, 0x47, 0x40, 0xad, 0xd0, 0x08, 0x7b, 0xab, 0x0d,
0x51, 0x13, 0xc1, 0xf9, 0x4d, 0x76, 0x89, 0x9f, 0xa0, 0x29, 0xa9, 0xe0, 0xac, 0x34, 0xd4, 0x0e,
0x3f, 0xb1, 0xb7, 0x8b, 0x21, 0x3e, 0xf3, 0x27, 0xfd, 0x0e, 0x14, 0xf0, 0x71, 0xb0, 0x40, 0x0f,
0x2f, 0xb2, 0x6c, 0x2c, 0x0f, 0x0a, 0xac, 0xd1, 0x99, 0x35, 0x81, 0xc3, 0x4e, 0x97, 0x54, 0x10,
0x41, 0x10, 0x1a, 0x5e, 0x63, 0x42, 0xd6, 0x69, 0xc4, 0x12, 0x3c, 0xd3, 0x93, 0x13, 0xc0, 0x11,
0xf3, 0x35, 0x80, 0xc8, 0xd7, 0x9a, 0x58, 0x62, 0x23, 0x7b, 0x38, 0xe3, 0x37, 0x5c, 0xbf, 0x12,
0x9d, 0x97, 0xf6, 0xba, 0xbb, 0xd2, 0x22, 0xda, 0x7e, 0x5c, 0x85, 0xf3, 0xea, 0xd8, 0x2b, 0x13,
0x54, 0x7f, 0x77, 0x27, 0x7c, 0xe9, 0x87, 0x74, 0x2e, 0xa9, 0x30, 0x83, 0xbc, 0xc2, 0x41, 0x14,
0x3a, 0xdd, 0x01, 0x55, 0x10, 0xa1, 0xfd, 0xcc, 0x73, 0x8e, 0x8d, 0x93, 0x61, 0x46, 0xd5, 0x15,
0x88, 0xf8, 0x9b, 0xc3, 0xa4, 0x79, 0x73, 0xc7, 0x94, 0xe7, 0x89, 0xa3, 0xc5, 0x09, 0xaa, 0x16,
0xe6, 0x5a, 0xed, 0xb1, 0xc8, 0x31, 0x09, 0x7f, 0xc9, 0xc0, 0x34, 0xb3, 0x18, 0x8d, 0x3e, 0x17,
0xd9, 0xeb, 0x5a, 0x3a, 0xe9, 0x0f, 0xfa, 0x58, 0x34, 0xce, 0x20, 0x43, 0x69, 0x3d, 0x7e, 0x18,
0xb7, 0x49, 0x2c, 0x48, 0x85, 0x47, 0x80, 0xe0, 0x69, 0xe9, 0x9d, 0x53, 0xb4, 0xb9, 0xea, 0x19,
0x05, 0x6c, 0xb6, 0xde, 0x31, 0x9f, 0x0e, 0xeb, 0x8e, 0x80, 0x99, 0x63, 0x10, 0xf6, 0x95, 0x1a,
0x6b, 0xce, 0xc0, 0xac, 0x5d, 0xd7, 0x74, 0x53, 0xd3, 0xa7, 0x24, 0x73, 0xcd, 0x72, 0x01, 0x1b,
0xa2, 0x26, 0x41, 0x31, 0x9a, 0xec, 0xd1, 0xfd, 0x83, 0x52, 0x91, 0x03, 0x9b, 0x68, 0x6b, 0x1c,
0xcc, 0x84, 0x37, 0x43, 0xf6, 0xa4, 0xab, 0x45, 0xde, 0x75, 0x2c, 0x13, 0x46, 0xec, 0xff, 0x1d,
0x7e, 0xa1, 0xad, 0xd5, 0x42, 0x7c, 0x25, 0x4e, 0x39, 0x1c, 0x28, 0x23, 0xe2, 0xa3, 0x80, 0x1e,
0x10, 0x03, 0xdb, 0xa7, 0x2e, 0x34, 0x5f, 0xf6, 0x64, 0x3b, 0x95, 0x33, 0x3f, 0x27, 0x14, 0x1f,
0x5e, 0xa7, 0xd8, 0x58, 0x1e, 0x14, 0x9b, 0x61, 0xf1, 0x6a, 0xc1, 0x45, 0x9c, 0xed, 0xa8, 0x20
};
//Первые 16 и вторые 16-итерационные ключи
// В оригинале они перевернуты. Потом подправить
__device__ unsigned char key[32] = {
0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10,
0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef
};
__constant__ unsigned char const_raws[16] = {
	148,32, 133, 16, 194, 192, 1, 251,
	1, 192, 194, 16, 133, 32, 148, 1 };
__device__ unsigned char key_mod_consts[32][16];
__device__ unsigned char round_keys[10][16];

__device__ unsigned char buffer_r_op[16];
__device__ unsigned char round_key_buff_1[16], round_key_buff_2[16], round_key_buff_3[16], round_key_buff_4[16];
__device__ unsigned char tmp_buff[16];

__device__ unsigned char IV[32];
//void print_consts();
//__device__ void calc_consts(int num_of_const, int num_of_thread);

__device__ void X_operation(unsigned char* a, unsigned char* b);
__device__ void S_operation(unsigned char* a);
__device__ unsigned char Galua_Mult(unsigned char a, unsigned char b);
__device__ void R_operation(unsigned char* a, unsigned char num_of_cell);
__device__ void L_operation(unsigned char* a, int num_of_cell);

__device__ void Feistel_operation(unsigned char* source_key_1, unsigned char* source_key_2,
	unsigned char* dest_key_1, unsigned char* dest_key_2, int thread, unsigned char* iter_const);
__device__ void Reverse_S_operation(unsigned char* a);
__device__ void Reverse_R_operation(unsigned char* a, int thread);
__device__ void Reverse_L_operation(unsigned char* a, int thread);

__device__ void generate_round_keys(int thread);
__device__ void print_round_keys();

__device__ void encript_block(unsigned char* data, int thread);
__device__ void decript_block(unsigned char* data, int thread);

__global__ void kuznechik(unsigned char* buffer, int size);

__global__ void host_print_block(unsigned char* block);
cudaError_t kuznechik_cuda(unsigned char* enc, int size);

__device__ void print_block(unsigned char* block);

__global__ void uni_key_generator();





void tests();
void run_hard_tests(int max_test_buff_size);

void clear_file()
{
	FILE* out = fopen("output.txt", "w");
	fclose(out);
}

void add_info_to_file(char* mode, int thread_num, int block_num, int size,unsigned int time)
{
	FILE* out = fopen("output.txt", "a");
	fprintf(out, "MODE: %s; Thread num: %d; Block num: %d; Buffer size: %d; Time: %u\n", mode, thread_num,
		block_num, size, time);
	fclose(out);
}

__global__ void host_print_block(unsigned char* block)
{
	int i = threadIdx.x;
	if (i == 0)
	{
		print_block(block);
	}
}

int main()
{
	tests();
	cout << "start" << endl;
	unsigned int start = clock();
	run_hard_tests(17);
	unsigned int end = clock();
	cout << endl << end - start << endl;
	return 0;
}




__global__ void uni_key_generator()
{
	int i = threadIdx.x;
	generate_round_keys(i);
}

__device__ void print_block(unsigned char* block)
{
	for (int i = 0; i < 16; i++)
	{
		printf("%x ", block[i]);
	}
	printf("\n");
}
__device__ void X_operation(unsigned char* a, unsigned char* b)
{
	(*a) = (*a) ^ (*b);
}
//S операция выполняется для всех сразу для блока в 16 байт. Меняется первый символ
__device__ void S_operation(unsigned char* a)
{

	(*a) = pi_table[*a];
}


__device__ unsigned char Galua_Mult(unsigned char a, unsigned char b)
{
	unsigned char c = 0;
	unsigned hi_bit;
	unsigned char tmp_a = a;
	unsigned char tmp_b = b;
	for (int i = 0; i < 8; i++)
	{
		if ((tmp_b & 1) == 1)
		{
			c ^= tmp_a;
		}
		hi_bit = tmp_a & 0x80;
		tmp_a = tmp_a << 1;
		if (hi_bit >= 128)
		{
			tmp_a ^= 0xc3;
		}
		tmp_b = tmp_b >> 1;
	}
	return c;
}
//num_of_cell-номер клетки/номер потока
__device__ void R_operation(unsigned char* a, unsigned char num_of_cell)
{
	__shared__ unsigned char unity_buffer[16];
	__shared__ unsigned char galua_multiple_buffer[16];
	unity_buffer[num_of_cell] = a[num_of_cell];
	galua_multiple_buffer[num_of_cell] = Galua_Mult(a[num_of_cell], const_raws[num_of_cell]);
	//buffer_r_op[num_of_cell] =  *(galua_multiple_calc+a[num_of_cell]+const_raws[num_of_cell]*256);
	//buffer_r_op[num_of_cell] = Galua_Mult(*a, num_of_cell);
	__syncthreads();
	if (num_of_cell % 2 == 0)
	{
		galua_multiple_buffer[num_of_cell] ^= galua_multiple_buffer[num_of_cell + 1];
	}
	__syncthreads();
	if (num_of_cell % 4 == 0)
	{
		galua_multiple_buffer[num_of_cell] ^= galua_multiple_buffer[num_of_cell + 2];
	}
	__syncthreads();
	if (num_of_cell % 8 == 0)
	{
		galua_multiple_buffer[num_of_cell] ^= galua_multiple_buffer[num_of_cell + 4];
	}
	__syncthreads();
	if (num_of_cell == 0)
	{
		galua_multiple_buffer[num_of_cell] ^= galua_multiple_buffer[num_of_cell + 8];
		a[num_of_cell] = galua_multiple_buffer[num_of_cell];
	}
	else
	{
		a[num_of_cell] = unity_buffer[num_of_cell - 1];
	}
}

__device__ void L_operation(unsigned char* a, int num_of_cell)
{

	for (int i = 0; i < 16; i++)
	{
		R_operation(a, num_of_cell);
		__syncthreads();

	}
}

__device__ void Feistel_operation(unsigned char* source_key_1, unsigned char* source_key_2,
	unsigned char* dest_key_1, unsigned char* dest_key_2, int thread, unsigned char* iter_const)
{

	dest_key_2[thread] = source_key_1[thread];
	dest_key_1[thread] = source_key_1[thread];
	X_operation(dest_key_1 + thread, iter_const + thread);
	S_operation(dest_key_1 + thread);
	__syncthreads();
	L_operation(dest_key_1, thread);
	X_operation(dest_key_1 + thread, source_key_2 + thread);
	__syncthreads();

}
__device__ void generate_round_keys(int thread)
{

	__syncthreads();
	round_keys[0][thread] = key[thread];
	round_keys[1][thread] = key[thread + 16];
	round_key_buff_1[thread] = key[thread];
	round_key_buff_2[thread] = key[thread + 16];

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			__syncthreads();
			Feistel_operation(round_key_buff_1, round_key_buff_2, round_key_buff_3, round_key_buff_4, thread,
				key_initial_consts + 16 * (8 * i + j * 2));
			__syncthreads();
			Feistel_operation(round_key_buff_3, round_key_buff_4, round_key_buff_1, round_key_buff_2, thread,
				key_initial_consts + 16 * (8 * i + j * 2 + 1));
			__syncthreads();
		}
		round_keys[2 * i + 2][thread] = round_key_buff_1[thread];
		round_keys[2 * i + 3][thread] = round_key_buff_2[thread];
	}

}

__device__ void print_round_keys()
{
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 16; j++)
		{
			printf("%x ", round_keys[i][j]);
		}
		printf("\n");
	}
}

__device__ void Reverse_S_operation(unsigned char* a)
{

	(*a) = reverse_pi_table[(*a)];

}

__device__ void Reverse_R_operation(unsigned char* a, int thread)
{
	unsigned char tmp;
	if (thread == 0)
	{
		tmp = a[15];
	}
	else
	{
		tmp = a[thread - 1];
		buffer_r_op[thread] = Galua_Mult(tmp, const_raws[thread]);
	}

	__syncthreads();
	if (thread == 0)
	{
		a[0] = tmp;
		for (int i = 1; i < 16; i++)
		{
			a[0] ^= buffer_r_op[i];
		}

	}
	else
	{
		a[thread] = tmp;
	}
	__syncthreads();
}

__device__ void Reverse_L_operation(unsigned char* a, int thread)
{

	for (int i = 0; i < 16; i++)
	{
		Reverse_R_operation(a, thread);
		__syncthreads();
	}

}

__device__ void encript_block(unsigned char* data, int thread)
{
	for (int j = 0; j < 9; j++)
	{
		X_operation(data + thread, round_keys[j] + thread);
		S_operation(data + thread);
		__syncthreads();
		L_operation(data, thread);
	}
	X_operation(data + thread, round_keys[9] + thread);
}
__device__ void decript_block(unsigned char* data, int thread)
{
	X_operation(data + thread, round_keys[9] + thread);
	for (int j = 8; j >= 0; j--)
	{
		__syncthreads();
		Reverse_L_operation(data, thread);
		__syncthreads();
		Reverse_S_operation(data + thread);

		X_operation(data + thread, round_keys[j] + thread);
	}
}


__device__ void encrypt_ECB(unsigned char* block, int thread)
{
	encript_block(block, thread);
}
__global__ void ECB_wrap(unsigned char* data, int size, int total_block_num)
{
	int i = threadIdx.x;
	int j = blockIdx.x;
	for (int k = j; k < (size / 16); k += total_block_num)
	{
		__syncthreads();
		encrypt_ECB(data + 16 * k, i);
	}

}
__global__ void CTR_wrap(unsigned char* block, int size, int total_block_num)
{
	int i = threadIdx.x;
	int j = blockIdx.x;
	__shared__ unsigned char work_buffer[16];
	for (long long k = j; k < size / 16; k += total_block_num)
	{
		work_buffer[i] = IV[i] + ((k >> (8 * (15-i))) & 0xff);
		__syncthreads();
		encrypt_ECB(work_buffer, i);
		__syncthreads();
		X_operation(block + k * 16+i, work_buffer+i);
	}
}
__global__ void encrypt_OFB(unsigned char* block, int size)
{
	int i = threadIdx.x;
	__shared__ unsigned char buff_1[16];
	__shared__ unsigned char buff_2[16];
	__shared__ unsigned char buff_3[16];
	buff_1[i] = IV[i];
	buff_2[i] = IV[i + 16];
	for (int k = 0; k < size / 16; k++)
	{
		__syncthreads();
		encrypt_ECB(buff_1, i);
		buff_3[i] = buff_2[i];
		buff_2[i] = buff_1[i];
		__syncthreads();
		X_operation(block + k * 16 + i, buff_1 + i);
		buff_1[i] = buff_3[i];
	}
}
__global__ void encrypt_CBC(unsigned char* block, int size)
{
	int i = threadIdx.x;
	__shared__ unsigned char buff_1[16];
	__shared__ unsigned char buff_2[16];
	buff_1[i] = IV[i];
	buff_2[i] = IV[i + 16];
	for (int k = 0; k < size / 16; k++)
	{
		__syncthreads();
		X_operation(block + k * 16 + i, buff_1 + i);
		__syncthreads();
		buff_1[i] = buff_2[i];
		encrypt_ECB(block + k * 16, i);
		__syncthreads();
		buff_2[i] = (block + k * 16)[i];
	}
}
__global__ void encrypt_CFB(unsigned char* block, int size)
{
	int i = threadIdx.x;
	__shared__ unsigned char buff_1[16];
	__shared__ unsigned char buff_2[16];
	buff_1[i] = IV[i];
	buff_2[i] = IV[i + 16];
	for (int k = 0; k < size / 16; k++)
	{
		__syncthreads();
		encrypt_ECB(buff_1, i);
		__syncthreads();
		X_operation(block + k * 16 + i, buff_1 + i);
		buff_1[i] = buff_2[i];
		buff_2[i] = (block + k * 16)[i];
	}
}





void run_hard_tests(int max_test_buff_size)
{
	clear_file();
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return;
	}
	cudaStatus = cudaMemcpyToSymbol(key, KEY_EXAMPLE, 32, 0, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return;
	}
	cudaStatus = cudaMemcpyToSymbol(IV, CIPHER_MODS::CBC_IV, 32, 0, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return;
	}
	uni_key_generator << <1, 16 >> > ();

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return;
	}
	int block_sizes[] = { 1, 16, 64, 128, 256, 512, 1024 };
	int one_mb_size = 1024 * 1024;
	for (int i = 0; i < 7; i++)
	{
		
		if (block_sizes[i] >= max_test_buff_size)
		{
			break;
		}
		cout << block_sizes[i] << " MB" << endl;
		int size = one_mb_size * block_sizes[i];


		unsigned char* original = new unsigned char[size];
		
		memset(original, 0, size);
		unsigned char* block;
		unsigned int start, end;
		start = clock();
		cudaStatus = cudaMalloc(&block, size);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc returned error code %d !\n", cudaStatus);
			fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
			return;
		}
		cudaMemcpy(block, original, size, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy returned error code %d !\n", cudaStatus);
			fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
			return;
		}
		end = clock();
		add_info_to_file("DATA COPY TO DEVICE", 1, 1, block_sizes[i], end - start);


		int block_num = 1024 * 1024 * 128;
		cout << "ECB " << 1 << " Block" << endl;
		start = clock();
		ECB_wrap << <1, 16 >> > (block, size, 1);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
			fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
			return;
		}
		end = clock();
		add_info_to_file("ECB", 16, 1, block_sizes[i], end - start);

		cout << "ECB " << block_num << " Block" << endl;
		start = clock();
		ECB_wrap << <block_num, 16 >> > (block, size, block_num);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
			fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
			return;
		}
		end = clock();
		add_info_to_file("ECB", 16, block_num, block_sizes[i], end - start);

		cout << "CTR " << 1 << " Block" << endl;
		start = clock();
		CTR_wrap << <1, 16 >> > (block, size, 1);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
			fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
			return;
		}
		end = clock();
		add_info_to_file("CTR", 16, 1, block_sizes[i], end - start);

		cout << "CTR " << block_num << " Block" << endl;
		start = clock();
		CTR_wrap << <block_num, 16 >> > (block, size, block_num);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
			fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
			return;
		}
		end = clock();
		add_info_to_file("CTR", 16, block_num, block_sizes[i], end - start);


		cout << "OFB " << 1 << " Block" << endl;
		start = clock();
		encrypt_OFB << <1, 16 >> > (block, size);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
			fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
			return;
		}
		end = clock();
		add_info_to_file("OFB", 16, 1, block_sizes[i], end - start);


		cout << "CBC " << 1 << " Block" << endl;
		start = clock();
		encrypt_CBC << <1, 16 >> > (block, size);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
			fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
			return;
		}
		end = clock();
		add_info_to_file("CBC", 16, 1, block_sizes[i], end - start);


		cout << "CFB " << 1 << " Block" << endl;
		start = clock();
		encrypt_CFB << <1, 16 >> > (block, size);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
			fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
			return;
		}
		end = clock();
		add_info_to_file("CFB", 16, 1, block_sizes[i], end - start);


		start = clock();
		cudaMemcpy(original, block, size, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy returned error code %d !\n", cudaStatus);
			fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
			return;
		}
		end = clock();
		add_info_to_file("DATA COPY FROM DEVICE", 1, 1, block_sizes[i], end - start);

		cudaFree(block);
		delete[] original;



	}
	
}


////tests_of_correct



__global__ void S_operation_test(unsigned char* test_array)
{
	int i = threadIdx.x;

	S_operation(test_array + i);

}
int test_s()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	unsigned char data[16];
	
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 0;
	}
	unsigned char* device_data;
	cudaStatus = cudaMalloc(&device_data, 16);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	cudaStatus = cudaMemcpy(device_data, S_TEST::S_1, 16, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	S_operation_test<<<1, 16>>>(device_data);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	cudaStatus = cudaMemcpy(data, device_data, 16, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	if (memcmp(data, S_TEST::S_2, 16) != 0)
	{
		printf("ERROR WHILE CHECKING S operation!\n");
		return 0;
	}
	S_operation_test << <1, 16 >> > (device_data);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	cudaStatus = cudaMemcpy(data, device_data, 16, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	if (memcmp(data, S_TEST::S_3, 16) != 0)
	{
		printf("ERROR WHILE CHECKING S operation!\n");
		return 0;
	}
	S_operation_test << <1, 16 >> > (device_data);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	cudaStatus = cudaMemcpy(data, device_data, 16, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	if (memcmp(data, S_TEST::S_4, 16) != 0)
	{
		printf("ERROR WHILE CHECKING S operation!\n");
		return 0;
	}
	S_operation_test << <1, 16 >> > (device_data);
	
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	cudaStatus = cudaMemcpy(data, device_data, 16, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	if (memcmp(data, S_TEST::S_RESULT, 16) != 0)
	{
		printf("ERROR WHILE CHECKING S operation!\n");
		return 0;
	}
	printf("S CORRECT\n");
	cudaFree(device_data);
	return 1;
}

__global__ void R_operation_test(unsigned char* test_array)
{
	int i = threadIdx.x;

	R_operation(test_array, i);

}
int test_r()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	unsigned char data[16];
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 0;
	}
	unsigned char* device_data;
	cudaStatus = cudaMalloc(&device_data, 16);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	cudaStatus = cudaMemcpy(device_data, R_TEST::R_1, 16, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	R_operation_test << <1, 16 >> > (device_data);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	cudaStatus = cudaMemcpy(data, device_data, 16, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	if (memcmp(data, R_TEST::R_2, 16) != 0)
	{
		printf("ERROR WHILE CHECKING R operation!\n");
		return 0;
	}
	R_operation_test << <1, 16 >> > (device_data);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	cudaStatus = cudaMemcpy(data, device_data, 16, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	if (memcmp(data, R_TEST::R_3, 16) != 0)
	{
		printf("ERROR WHILE CHECKING R operation!\n");
		return 0;
	}
	R_operation_test << <1, 16 >> > (device_data);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	cudaStatus = cudaMemcpy(data, device_data, 16, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	if (memcmp(data, R_TEST::R_4, 16) != 0)
	{
		printf("ERROR WHILE CHECKING R operation!\n");
		return 0;
	}
	R_operation_test << <1, 16 >> > (device_data);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	cudaStatus = cudaMemcpy(data, device_data, 16, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	if (memcmp(data, R_TEST::R_RESULT, 16) != 0)
	{
		printf("ERROR WHILE CHECKING R operation!\n");
		return 0;
	}
	printf("R CORRECT\n");
	cudaFree(device_data);
	return 1;
}



__global__ void L_operation_test(unsigned char* test_array)
{
	int i = threadIdx.x;
	L_operation(test_array, i);

}
int test_l()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	unsigned char data[16];
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 0;
	}
	unsigned char* device_data;
	cudaStatus = cudaMalloc(&device_data, 16);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	cudaStatus = cudaMemcpy(device_data, L_TEST::L_1, 16, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	L_operation_test << <1, 16 >> > (device_data);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	cudaStatus = cudaMemcpy(data, device_data, 16, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	if (memcmp(data, L_TEST::L_2, 16) != 0)
	{
		printf("ERROR WHILE CHECKING L operation!\n");
		return 0;
	}
	L_operation_test << <1, 16 >> > (device_data);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	cudaStatus = cudaMemcpy(data, device_data, 16, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	if (memcmp(data, L_TEST::L_3, 16) != 0)
	{
		printf("ERROR WHILE CHECKING L operation!\n");
		return 0;
	}
	L_operation_test << <1, 16 >> > (device_data);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	cudaStatus = cudaMemcpy(data, device_data, 16, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	if (memcmp(data, L_TEST::L_4, 16) != 0)
	{
		printf("ERROR WHILE CHECKING L operation!\n");
		return 0;
	}
	L_operation_test << <1, 16 >> > (device_data);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	cudaStatus = cudaMemcpy(data, device_data, 16, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	if (memcmp(data, L_TEST::L_RESULT, 16) != 0)
	{
		printf("ERROR WHILE CHECKING L operation!\n");
		return 0;
	}
	printf("L CORRECT\n");
	cudaFree(device_data);
	return 1;
}


int test_keys()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	unsigned char data[16];
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 0;
	}
	cudaStatus = cudaMemcpyToSymbol(key, KEY_EXAMPLE, 32, 0, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	uni_key_generator<<<1,16>>>();
	unsigned char output_keys[16];
	for (int i = 0; i < 10; i++)
	{
		cudaStatus = cudaMemcpyFromSymbol(output_keys, round_keys, 16,16*i, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy returned error code %d while copying %d block!\n", cudaStatus, i);
			fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
			return 0;
		}
		if (memcmp(output_keys, KEY_TEST+i * 16, 16) != 0)
		{
			printf("ERROR WHILE CHECKING KEYS!\n");
			return 0;
		}
	}
	printf("KEYS CORRECT\n");
	return 1;
}

__global__ void test_simple_block(unsigned char* block)
{
	int i = threadIdx.x;
	encript_block(block, i);
}
int test_only_block()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	unsigned char data[16];
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 0;
	}
	cudaStatus = cudaMemcpyToSymbol(key, KEY_EXAMPLE, 32, 0, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	uni_key_generator<<<1,16>>>();

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	unsigned char* block;
	cudaStatus = cudaMalloc(&block, 16);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	cudaStatus = cudaMemcpy(block, SIMPLE_BLOCK::OPEN_TEXT, 16, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	test_simple_block<<<1,16>>>(block);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	
	cudaStatus = cudaMemcpy(data, block, 16, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	cudaFree(block);
	if (memcmp(data, SIMPLE_BLOCK::CIPHER, 16) != 0)
	{
		printf("SIMPLE BLOCK ERROR \n");
		return 0;
	}
	printf("SIMPLE BLOCK Correct \n");
	return 1;
}

int test_modes()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	unsigned char data[64];
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 0;
	}
	cudaStatus = cudaMemcpyToSymbol(key, KEY_EXAMPLE, 32, 0, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	uni_key_generator << <1, 16 >> > ();

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	unsigned char* block;
	cudaStatus = cudaMalloc(&block, CIPHER_MODS::DATA_SIZE);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	int block_num = 3;
	for (block_num = 1; block_num < 6; block_num++)
	{
		cudaStatus = cudaMemcpy(block, CIPHER_MODS::OPEN_TEXT, CIPHER_MODS::DATA_SIZE, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy returned error code %d !\n", cudaStatus);
			fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
			return 0;
		}

		ECB_wrap << <block_num, 16 >> > (block, CIPHER_MODS::DATA_SIZE, block_num);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
			fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
			return 0;
		}

		cudaStatus = cudaMemcpy(data, block, CIPHER_MODS::DATA_SIZE, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy returned error code %d !\n", cudaStatus);
			fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
			return 0;
		}

		if (memcmp(data, CIPHER_MODS::ECB, CIPHER_MODS::DATA_SIZE) != 0)
		{
			printf("ECB ERROR %d\n", block_num);
			return 0;
		}
		printf("ECB CORRECT %d\n", block_num);


		cudaStatus = cudaMemcpy(block, CIPHER_MODS::OPEN_TEXT, CIPHER_MODS::DATA_SIZE, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy returned error code %d !\n", cudaStatus);
			fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
			return 0;
		}
		cudaStatus = cudaMemcpyToSymbol(IV, CIPHER_MODS::CTR_IV, 8, 0, cudaMemcpyHostToDevice);
		CTR_wrap << <block_num, 16 >> > (block, CIPHER_MODS::DATA_SIZE, block_num);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
			fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
			return 0;
		}

		cudaStatus = cudaMemcpy(data, block, CIPHER_MODS::DATA_SIZE, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy returned error code %d !\n", cudaStatus);
			fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
			return 0;
		}

		if (memcmp(data, CIPHER_MODS::CTR, CIPHER_MODS::DATA_SIZE) != 0)
		{
			printf("CTR ERROR %d\n", block_num);
			return 0;
		}
		printf("CTR CORRECT %d\n", block_num);
	}



	cudaStatus = cudaMemcpy(block, CIPHER_MODS::OPEN_TEXT, CIPHER_MODS::DATA_SIZE, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	cudaStatus = cudaMemcpyToSymbol(IV, CIPHER_MODS::OFB_IV, 32, 0, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	encrypt_OFB << <1, 16 >> > (block, CIPHER_MODS::DATA_SIZE);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}

	cudaStatus = cudaMemcpy(data, block, CIPHER_MODS::DATA_SIZE, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}

	if (memcmp(data, CIPHER_MODS::OFB, CIPHER_MODS::DATA_SIZE) != 0)
	{
		printf("OFB ERROR\n");
		return 0;
	}
	printf("OFB CORRECT\n");


	cudaStatus = cudaMemcpy(block, CIPHER_MODS::OPEN_TEXT, CIPHER_MODS::DATA_SIZE, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	cudaStatus = cudaMemcpyToSymbol(IV, CIPHER_MODS::CBC_IV, 32, 0, cudaMemcpyHostToDevice);
	encrypt_CBC << <1, 16 >> > (block, CIPHER_MODS::DATA_SIZE);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}

	cudaStatus = cudaMemcpy(data, block, CIPHER_MODS::DATA_SIZE, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}

	if (memcmp(data, CIPHER_MODS::CBC, CIPHER_MODS::DATA_SIZE) != 0)
	{
		printf("CBC ERROR\n");
		return 0;
	}
	printf("CBC CORRECT\n");


	cudaStatus = cudaMemcpy(block, CIPHER_MODS::OPEN_TEXT, CIPHER_MODS::DATA_SIZE, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}
	cudaStatus = cudaMemcpyToSymbol(IV, CIPHER_MODS::CFB_IV, 8, 0, cudaMemcpyHostToDevice);
	encrypt_CFB << <1, 16 >> > (block, CIPHER_MODS::DATA_SIZE);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}

	cudaStatus = cudaMemcpy(data, block, CIPHER_MODS::DATA_SIZE, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy returned error code %d !\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}

	if (memcmp(data, CIPHER_MODS::CFB, CIPHER_MODS::DATA_SIZE) != 0)
	{
		printf("CFB ERROR\n");
		return 0;
	}
	printf("CFB CORRECT\n");



	cudaFree(block);
	
	return 1;
}

void tests()
{
	if (!test_s())
	{
		return;
	}
	
	if (!test_r())
	{
		return;
	}
	if (!test_l())
	{
		return;
	}
	if (!test_keys())
	{
		return;
	}
	if (!test_only_block())
	{
		return;
	}
	if (!test_modes())
	{
		return;
	}
	return;
}