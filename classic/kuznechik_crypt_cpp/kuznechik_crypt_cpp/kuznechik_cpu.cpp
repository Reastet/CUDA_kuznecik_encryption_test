#include "kuznechik_cpu.h"
void kuznechik_cpu::print_consts()
{
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < 16; j++)
		{
			cout << hex <<(int) this->key_mod_consts[i][j] << " ";
		}
		cout << endl << endl;
	}
}
void kuznechik_cpu::calc_consts()
{
	for (int i = 0; i < 32; i++)
	{
		
		for (int j = 0; j < 16; j++)
		{
			this->key_mod_consts[i][j] = 0;
		}
		this->key_mod_consts[i][15] = i+1;
		this->L_operation(this->key_mod_consts[i]);
	}
}


void kuznechik_cpu::X_operation(unsigned char* a, unsigned char* b)
{
	for (int i = 0; i < 16; i++)
	{
		a[i] ^= b[i];
	}
}

void kuznechik_cpu::S_operation(unsigned char* a)
{
	for (int i = 0; i < 16; i++)
	{
		a[i] = this->pi_table[a[i] / 16][a[i] % 16];
	}
}

unsigned char kuznechik_cpu::Galua_Mult(unsigned char a, unsigned char b)
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

void kuznechik_cpu::R_operation(unsigned char* a)
{
	unsigned char a_summ = 0;
	unsigned char res[16];
	for (int i = 15; i > 0; i--)
	{
		a_summ ^= this->Galua_Mult(a[i], this->const_raws[i]);
		a[i] = a[i - 1];

		
	}
	a_summ ^= this->Galua_Mult(a[0], this->const_raws[0]);

	a[0] = a_summ;

}

void kuznechik_cpu::L_operation(unsigned char* a)
{
	for (int i = 0; i < 16; i++)
	{
		this->R_operation(a);
	}
}

void kuznechik_cpu::Feistel_operation(unsigned char* key_a, unsigned char* key_b, unsigned char* iter_const)
{
	unsigned char tmp[16];
	for (int i = 0; i < 16; i++)
	{
		tmp[i] = key_a[i];
	}
	this->X_operation(key_a, iter_const);
	this->S_operation(key_a);
	this->L_operation(key_a);
	this->X_operation(key_a, key_b);
	for (int i = 0; i < 16; i++)
	{
		key_b[i] = tmp[i];
	}
}
void kuznechik_cpu::generate_round_keys()
{
	unsigned char key_1[16], key_2[16], key_3[16], key_4[16];

	for (int i = 0; i < 16; i++)
	{
		this->round_keys[0][i] = this->key[i];
		key_1[i] = this->key[i];
		this->round_keys[1][i] = this->key[i+16];
		key_2[i] = this->key[i + 16];
	}
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			this->Feistel_operation(key_1, key_2, this->key_mod_consts[8*i+j*2]);
			for (int k = 0; k < 16; k++)
			{
				key_3[k] = key_1[k];
				key_4[k] = key_2[k];
			}
			this->Feistel_operation(key_3, key_4, this->key_mod_consts[8 * i + j * 2+1]);
			for (int k = 0; k < 16; k++)
			{
				key_1[k] = key_3[k];
				key_2[k] = key_4[k];
			}
		}
		for (int j = 0; j < 16; j++)
		{
			this->round_keys[2 * i + 2][j] = key_1[j];
			this->round_keys[2 * i + 3][j] = key_2[j];
		}

	}
	
}

void kuznechik_cpu::print_round_keys()
{
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 16; j++)
		{
			cout << hex << (int)this->round_keys[i][j] << " ";
		}
		cout << endl << endl;
	}
}

unsigned char* kuznechik_cpu::Reverse_S_operation(unsigned char* a)
{
	unsigned char* res = new unsigned char[16];
	for (int i = 0; i < 16; i++)
	{
		unsigned char tmp = this->reverse_pi_table[a[i] / 16][a[i] % 16];
		res[i] = this->reverse_pi_table[a[i] / 16][a[i] % 16];
	}
	return res;
}

void kuznechik_cpu::Reverse_R_operation(unsigned char* a)
{
	unsigned char a_0 = a[15];
	unsigned char* res = new unsigned char[16];
	for (int i = 1; i < 16; i++)
	{
		res[i] = a[i - 1];
		a_0 ^= this->Galua_Mult(res[i], this->const_raws[i]);
		
	}
	res[0] = a_0;
	for (int i = 0; i < 16; i++)
	{
		a[i] = res[i];
	}
	delete[] res;
}

unsigned char* kuznechik_cpu::Reverse_L_operation(unsigned char* a)
{
	unsigned char* res = new unsigned char[16];
	for (int i = 0; i < 16; i++)
	{
		res[i] = a[i];
	}
	for (int i = 0; i < 16; i++)
	{
		this->Reverse_R_operation(res);
	}
	return res;
}

void kuznechik_cpu::encript_simple(unsigned char* data)
{
	for (int j = 0; j < 9; j++)
	{
		this->X_operation(data, this->round_keys[j]);
		this->S_operation(data);
		this->L_operation(data);
	}
	this->X_operation(data, this->round_keys[9]);
}
unsigned char* kuznechik_cpu::decript(unsigned char* data, int size)
{
	int t = 0;
	if (size % 16 != 0)
	{
		t = 1;
	}
	int size_out = ((int)(size / 16) + t) * 16;
	unsigned char* out = new unsigned char[size_out];

	for (int i = 0; i < size_out; i++)
	{
		out[i] = 0;
		if (i >= size)
		{
			out[i] = 0;
		}
		else
		{
			out[i] = data[i];
		}
	}

	for (int i = 0; i < size_out / 16; i++)
	{
		unsigned char* tmp = NULL;
		this->X_operation(out + (16 * i), this->round_keys[9]);
		for (int k = 0; k < 16; k++)
		{
			out[k + (16 * i)] = tmp[k];
		}
		delete[] tmp;
		for (int j = 8; j >=0; j--)
		{
			tmp = this->Reverse_L_operation(out + (16 * i));
			for (int k = 0; k < 16; k++)
			{
				out[k + (16 * i)] = tmp[k];
			}
			delete[] tmp;
			
			tmp = this->Reverse_S_operation(out + (16 * i));
			for (int k = 0; k < 16; k++)
			{
				out[k + (16 * i)] = tmp[k];
			}
			delete[] tmp;
			this->X_operation(out + (16 * i), this->round_keys[j]);
			for (int k = 0; k < 16; k++)
			{
				out[k + (16 * i)] = tmp[k];
			}
			delete[] tmp;
		}
		
	}
	return out;

}


bool kuznechik_cpu::saveHexToFile(const unsigned char* data, size_t size,
	const std::string& filename, bool append) {
	std::ios_base::openmode mode = std::ios::out;

	if (append) {
		mode |= std::ios::app;  // Режим добавления в конец файла
	}
	else {
		mode |= std::ios::trunc;  // Режим перезаписи (создаёт новый файл)
	}

	std::ofstream file(filename, mode);

	if (!file.is_open()) {
		std::cerr << "Ошибка: не удалось открыть файл для записи: " << filename << std::endl;
		return false;
	}
	// Если дописываем в конец, добавляем перенос строки перед новыми данными
	if (append) {
		file << "\n\n";
	}

	// Устанавливаем формат вывода: шестнадцатеричный, с ведущими нулями (2 цифры на байт)
	file << std::hex << std::setfill('0');

	for (size_t i = 0; i < size; ++i) {
		// Выводим каждый байт как двузначное hex-число
		file << "0x" << std::setw(2) << static_cast<int>(data[i]) << ", ";
	}

	file.close();

	if (file.fail()) {
		std::cerr << "Ошибка: запись в файл завершилась с ошибкой: " << filename << std::endl;
		return false;
	}

	return true;
}

void kuznechik_cpu::save_consts_to_files(const std::string& filename)
{
	unsigned char data[16];
	for (int j = 0; j < 16; j++)
	{
		data[j] = this->key_mod_consts[0][j];
	}
	this->saveHexToFile(data, 16, filename);
	
	for (int i = 1; i < 32; i++)
	{
		for (int j = 0; j < 16; j++)
		{
			data[j] = this->key_mod_consts[i][j];
		}
		this->saveHexToFile(data, 16, filename, true);
	}

}

void kuznechik_cpu::encript_ECB(unsigned char* data, int size)
{
	if (size % 16 != 0)
	{
		printf("SIZE \% 16 != 0\n");
	}
	for (int i = 0; i < (size / 16); i++)
	{
		this->encript_simple(data+i*16);
	}
}

void kuznechik_cpu::encript_CTR(unsigned char* data, unsigned char* IV, int size)
{
	if (size % 16 != 0)
	{
		printf("SIZE \% 16 != 0\n");
	}
	unsigned char IV_BUF[16], IV_CTR[16];
	memcpy(IV_CTR, IV, 8);
	memset(IV_CTR + 8, 0, 8);
	for (int i = 0; i < size / 16; i++)
	{
		memcpy(IV_BUF, IV_CTR, 16);
		IV_CTR[15]++;
		if (IV_CTR[15] == 0)
		{
			for (int i = 14; i > 0; i--)
			{
				IV_CTR[i]++;
				if (IV_CTR[i] != 0)
				{
					break;
				}
			}
		}
		this->encript_simple(IV_BUF);
		this->X_operation(data + i * 16, IV_BUF);
	}

}


void kuznechik_cpu::encript_OFB(unsigned char* data, unsigned char* IV, int size)
{
	if (size % 16 != 0)
	{
		printf("SIZE \% 16 != 0\n");
	}
	unsigned char IV_BUF_1[16], IV_BUF_2[16], TMP[16];
	memcpy(IV_BUF_1, IV, 16);
	memcpy(IV_BUF_2 ,IV+16, 16);
	for (int i = 0; i < size / 16; i++)
	{
		this->encript_simple(IV_BUF_1);
		this->X_operation(data + i * 16, IV_BUF_1);
		memcpy(TMP, IV_BUF_1, 16);
		memcpy(IV_BUF_1, IV_BUF_2, 16);
		memcpy(IV_BUF_2, TMP, 16);
	}
}


void kuznechik_cpu::encript_CBC(unsigned char* data, unsigned char* IV, int size)
{
	if (size % 16 != 0)
	{
		printf("SIZE \% 16 != 0\n");
	}
	unsigned char IV_BUF_1[16], IV_BUF_2[16];
	memcpy(IV_BUF_1, IV, 16);
	memcpy(IV_BUF_2, IV + 16, 16);
	for (int i = 0; i < size / 16; i++)
	{
		this->X_operation(data + i * 16, IV_BUF_1);
		this->encript_simple(data + i * 16);

		memcpy(IV_BUF_1, IV_BUF_2, 16);
		memcpy(IV_BUF_2, data + i * 16, 16);
	}
}


void kuznechik_cpu::encript_CFB(unsigned char* data, unsigned char* IV, int size)
{
	if (size % 16 != 0)
	{
		printf("SIZE \% 16 != 0\n");
	}
	unsigned char IV_BUF_1[16], IV_BUF_2[16];
	memcpy(IV_BUF_1, IV, 16);
	memcpy(IV_BUF_2, IV + 16, 16);
	for (int i = 0; i < size / 16; i++)
	{
		this->encript_simple(IV_BUF_1);
		this->X_operation(data + i * 16, IV_BUF_1);
		memcpy(IV_BUF_1, IV_BUF_2, 16);
		memcpy(IV_BUF_2, data + i * 16, 16);
	}
}



int kuznechik_cpu::test_s() 
{
	unsigned char work_buff[16];
	memcpy(work_buff, S_TEST::S_1, 16);
	this->S_operation(work_buff);
	if (memcmp(work_buff, S_TEST::S_2, 16))
	{
		printf("S_1 OPERATION CHECK ERROR!!!!\n");
		return 0;
	}
	this->S_operation(work_buff);
	if (memcmp(work_buff, S_TEST::S_3, 16))
	{
		printf("S_2 OPERATION CHECK ERROR!!!!\n");
		return 0;
	}
	this->S_operation(work_buff);
	if (memcmp(work_buff, S_TEST::S_4, 16))
	{
		printf("S_3 OPERATION CHECK ERROR!!!!\n");
		return 0;
	}
	this->S_operation(work_buff);
	if (memcmp(work_buff, S_TEST::S_RESULT, 16))
	{
		printf("S_4 OPERATION CHECK ERROR!!!!\n");
		return 0;
	}
	printf("S OPERATION CORRECT\n");
	return 1;
}
int kuznechik_cpu::test_r()
{
	unsigned char work_buff[16];
	memcpy(work_buff, R_TEST::R_1, 16);
	this->R_operation(work_buff);
	if (memcmp(work_buff, R_TEST::R_2, 16))
	{
		printf("R_1 OPERATION CHECK ERROR!!!!\n");
		return 0;
	}
	this->R_operation(work_buff);
	if (memcmp(work_buff, R_TEST::R_3, 16))
	{
		printf("R_2 OPERATION CHECK ERROR!!!!\n");
		return 0;
	}
	this->R_operation(work_buff);
	if (memcmp(work_buff, R_TEST::R_4, 16))
	{
		printf("R_3 OPERATION CHECK ERROR!!!!\n");
		return 0;
	}
	this->R_operation(work_buff);
	if (memcmp(work_buff, R_TEST::R_RESULT, 16))
	{
		printf("R_4 OPERATION CHECK ERROR!!!!\n");
		return 0;
	}
	printf("R OPERATION CORRECT\n");
	return 1;
}
int kuznechik_cpu::test_l()
{
	unsigned char work_buff[16];
	memcpy(work_buff, L_TEST::L_1, 16);
	this->L_operation(work_buff);
	if (memcmp(work_buff, L_TEST::L_2, 16))
	{
		printf("L_1 OPERATION CHECK ERROR!!!!\n");
		return 0;
	}
	this->L_operation(work_buff);
	if (memcmp(work_buff, L_TEST::L_3, 16))
	{
		printf("L_2 OPERATION CHECK ERROR!!!!\n");
		return 0;
	}
	this->L_operation(work_buff);
	if (memcmp(work_buff, L_TEST::L_4, 16))
	{
		printf("L_3 OPERATION CHECK ERROR!!!!\n");
		return 0;
	}
	this->L_operation(work_buff);
	if (memcmp(work_buff, L_TEST::L_RESULT, 16))
	{
		printf("L_4 OPERATION CHECK ERROR!!!!\n");
		return 0;
	}
	printf("L OPERATION CORRECT\n");
	return 1;
}
int kuznechik_cpu::test_keys()
{
	for (int i = 0; i < 32; i++) { this->key[i] = KEY_EXAMPLE[i]; };
	this->calc_consts();
	this->generate_round_keys();
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 16; j++)
		{
			if (this->round_keys[i][j] != KEY_TEST[i * 16 + j])
			{
				printf("ERROR IN KEY %d\n", i+1);
				return 0;
			}
		}
	}
	printf("KEYS CORRECT\n");
	return 1;

}
int kuznechik_cpu::test_one_block()
{
	for (int i = 0; i < 32; i++) { this->key[i] = KEY_EXAMPLE[i]; };
	this->calc_consts();
	this->generate_round_keys();
	unsigned char work_buff[16];
	memcpy(work_buff, SIMPLE_BLOCK::OPEN_TEXT, 16);
	this->encript_simple(work_buff);
	if (memcmp(work_buff, SIMPLE_BLOCK::CIPHER, 16))
	{
		printf("ENCRIPTION GOST 34.12-15 INCORRECT\n");
		return 0;
	}
	printf("ENCRIPTION GOST 34.12-15 CORRECT\n");
	return 1;
}

int kuznechik_cpu::test_ECB()
{
	for (int i = 0; i < 32; i++) { this->key[i] = KEY_EXAMPLE[i]; };
	this->calc_consts();
	this->generate_round_keys();
	unsigned char* encrypt = new unsigned char[CIPHER_MODS::DATA_SIZE];
	for (int i = 0; i < CIPHER_MODS::DATA_SIZE; i++) { encrypt[i] = CIPHER_MODS::OPEN_TEXT[i]; }
	this->encript_ECB(encrypt, CIPHER_MODS::DATA_SIZE);
	if (memcmp(encrypt, CIPHER_MODS::ECB, CIPHER_MODS::DATA_SIZE))
	{
		printf("ECB ERROR\n");
		return 0;
	}
	printf("ECB CORRECT\n");
	return 1;
}

int kuznechik_cpu::test_CTR()
{
	for (int i = 0; i < 32; i++) { this->key[i] = KEY_EXAMPLE[i]; };
	this->calc_consts();
	this->generate_round_keys();
	unsigned char* encrypt = new unsigned char[CIPHER_MODS::DATA_SIZE];
	unsigned char IV_TEST[8];
	for (int i = 0; i < CIPHER_MODS::DATA_SIZE; i++) { encrypt[i] = CIPHER_MODS::OPEN_TEXT[i]; }
	for (int i = 0; i < 8; i++) { IV_TEST[i] = CIPHER_MODS::CTR_IV[i]; }
	this->encript_CTR(encrypt,IV_TEST, CIPHER_MODS::DATA_SIZE);
	if (memcmp(encrypt, CIPHER_MODS::CTR, CIPHER_MODS::DATA_SIZE))
	{
		printf("CTR ERROR\n");
		return 0;
	}
	printf("CTR CORRECT\n");
	return 1;
}

int kuznechik_cpu::test_OFB()
{
	for (int i = 0; i < 32; i++) { this->key[i] = KEY_EXAMPLE[i]; };
	this->calc_consts();
	this->generate_round_keys();
	unsigned char* encrypt = new unsigned char[CIPHER_MODS::DATA_SIZE];
	unsigned char IV_TEST[32];
	for (int i = 0; i < CIPHER_MODS::DATA_SIZE; i++) { encrypt[i] = CIPHER_MODS::OPEN_TEXT[i]; }
	for (int i = 0; i < 32; i++) { IV_TEST[i] = CIPHER_MODS::OFB_IV[i]; }
	this->encript_OFB(encrypt, IV_TEST, CIPHER_MODS::DATA_SIZE);
	if (memcmp(encrypt, CIPHER_MODS::OFB, CIPHER_MODS::DATA_SIZE))
	{
		printf("OFB ERROR\n");
		return 0;
	}
	printf("OFB CORRECT\n");
	return 1;
}

int kuznechik_cpu::test_CBC()
{
	for (int i = 0; i < 32; i++) { this->key[i] = KEY_EXAMPLE[i]; };
	this->calc_consts();
	this->generate_round_keys();
	unsigned char* encrypt = new unsigned char[CIPHER_MODS::DATA_SIZE];
	unsigned char IV_TEST[32];
	for (int i = 0; i < CIPHER_MODS::DATA_SIZE; i++) { encrypt[i] = CIPHER_MODS::OPEN_TEXT[i]; }
	for (int i = 0; i < 32; i++) { IV_TEST[i] = CIPHER_MODS::CBC_IV[i]; }
	this->encript_CBC(encrypt, IV_TEST, CIPHER_MODS::DATA_SIZE);
	if (memcmp(encrypt, CIPHER_MODS::CBC, CIPHER_MODS::DATA_SIZE))
	{
		printf("CBC ERROR\n");
		return 0;
	}
	printf("CBC CORRECT\n");
	return 1;
}


int kuznechik_cpu::test_CFB()
{
	for (int i = 0; i < 32; i++) { this->key[i] = KEY_EXAMPLE[i]; };
	this->calc_consts();
	this->generate_round_keys();
	unsigned char* encrypt = new unsigned char[CIPHER_MODS::DATA_SIZE];
	unsigned char IV_TEST[32];
	for (int i = 0; i < CIPHER_MODS::DATA_SIZE; i++) { encrypt[i] = CIPHER_MODS::OPEN_TEXT[i]; }
	for (int i = 0; i < 32; i++) { IV_TEST[i] = CIPHER_MODS::CFB_IV[i]; }
	this->encript_CFB(encrypt, IV_TEST, CIPHER_MODS::DATA_SIZE);
	if (memcmp(encrypt, CIPHER_MODS::CFB, CIPHER_MODS::DATA_SIZE))
	{
		printf("CFB ERROR\n");
		return 0;
	}
	printf("CFB CORRECT\n");
	return 1;
}

void kuznechik_cpu::test_all()
{
	if (!this->test_s())
	{
		return;
	}
	if (!this->test_r())
	{
		return;
	}
	if (!this->test_l())
	{
		return;
	}
	if (!this->test_keys())
	{
		return;
	}
	if (!this->test_one_block())
	{
		return;
	}
	if (!this->test_ECB())
	{
		return;
	}
	if (!this->test_CTR())
	{
		return;
	}
	if (!this->test_OFB())
	{
		return;
	}
	if (!this->test_CBC())
	{
		return;
	}
	if (!this->test_CFB())
	{
		return;
	}
	return;
}