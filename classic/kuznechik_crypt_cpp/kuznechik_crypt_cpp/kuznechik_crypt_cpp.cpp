#include <iostream>
#include <ctime>
#include <string.h>
#include "kuznechik_cpu.h"

#include <thread>
#include <string>

using namespace std;
void thread_run_data(int mb_size, int mode)
{
    kuznechik_cpu provider;
    int size = 1024 * 1024* mb_size;
    unsigned char IV[32];
    memcpy(IV, CIPHER_MODS::CBC_IV, 32);
    unsigned char* data = new unsigned char[size];
    unsigned int start, end;
    switch (mode)
    {
    case 0:
    {
        start = clock();
        provider.encript_ECB(data, size);
        end = clock();
        break;
    }
    case 1:
    {
        start = clock();
        provider.encript_CTR(data, IV, size);
        end = clock();
        break;
    }
    case 2:
    {
        start = clock();
        provider.encript_OFB(data, IV, size);
        end = clock();
        break;
    }
    case 3:
    {
        start = clock();
        provider.encript_CBC(data, IV, size);
        end = clock();
        break;
    }
    case 4:
    {
        start = clock();
        provider.encript_CFB(data, IV, size);
        end = clock();
        break;
    }
    default:
    {
        return;
    }
    }
    //cout << end - start << endl;
    string out_size = to_string(mb_size);
    string out_mode = to_string(mode);
    ofstream output_data("output_data\\"+out_size + " size " + out_mode + " mode" + ".txt");
    output_data << end - start << endl;
    output_data.close();
    return;

}


int main()
{
    int tests[] = { 1, 16, 64, 128, 256, 512 };
    //thread_run_data(64, 0);
    /*for (int i = 0; i < 6; i++)
    {
        cout << tests[i] << endl;
        thread ecb(thread_run_data, tests[i], 0);
        thread ctr(thread_run_data, tests[i], 1);
        thread ofb(thread_run_data, tests[i], 2);
        thread cbc(thread_run_data, tests[i], 3);
        thread cfb(thread_run_data, tests[i], 4);
        ecb.join();
        ctr.join();
        ofb.join();
        cbc.join();
        cfb.join();
    }
    return 0;*/
    //thread my_thread;
    
    kuznechik_cpu elem;
    elem.test_all();
  //  elem.save_consts_to_files("output.txt");

 //   return 0;
 /*   unsigned char test[17] = {
        0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
        0x00, 0x77, 0x66, 0x55,0x44,0x33,0x22,0x11,
    0xff};*/
    int size = 1024 * 1024 * 16;
    elem.calc_consts();
    elem.generate_round_keys();
    unsigned char* control_test = new unsigned char[size];
  //  unsigned char IV[32];

    unsigned int start = clock();
    elem.encript_ECB(control_test, size);
  //  cout << start << endl;
    
 //   unsigned char control_test[TEXT_SIZE];
 //   memcpy(control_test, OPEN_TEXT, TEXT_SIZE);
 //   kuznechik_cpu elem;
  //  elem.calc_consts();
  //  elem.print_consts();
  // elem.save_consts_to_files("output.txt");
  //  elem.generate_round_keys();
 //   elem.print_round_keys();
   // return 0;
   // elem.encrypt_5_3_mode(control_test, IV, 1024 * 1024 * 128);
  /*  for (int i = 0; i < 16; i++)
    {

      //  cout << hex << (int)result[i] << " ";


    }
    if (memcmp(result, ECB_EXAMPLE, TEXT_SIZE) == 0)
    {
        cout << "OK" << endl;
    }
    else
    {
        cout << "NOT OK" << endl;
    }
    return 0;*/
   // 
  //  unsigned char* t_1;
  //  unsigned char* t_2;
  //  unsigned char IV[8] = { 0x12, 0x34, 0x56, 0x78, 0x90, 0xab, 0xce, 0xf0 };
    //t_1 = elem.encript_CTR(control_test, IV, size);
    unsigned int end = clock();
    cout << end<<" " << end-start << endl;
    return 0;
 
  //  unsigned int end = clock();
   // cout << end<<" " << end-start << endl;
}
