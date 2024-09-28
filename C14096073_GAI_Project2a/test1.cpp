
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <time.h>
using namespace std;

int main() {

    FILE* fpt;

    fpt = fopen("test1.csv", "w+");
    fprintf(fpt, "src, tgt\n");

    srand(time(NULL));
    int a, b, c, d, e;
    int op1, op2;
    string l, r;

 for (int i = 0; i < 10000; i++) 
 {
        a = rand() % 10;
        b = rand() % 10;
        c = rand() % 10;
        op1 = rand() % 2;
        op2 = rand() % 2;


        l = to_string(a);
        if (op1 == 0) {
            l += "+";
            d = a + b;
        }
        else {
            l += "-";
            d = a - b;
        }
        l += to_string(b);
        if (op2 == 0) {
            l += "+";
            d = d + c;
        }
        else {
            l += "-";
            d = d - c;
        }
        l += to_string(c);
        l += "=";
        r = to_string(d);
        fprintf(fpt, "%s, %s\n", l.c_str(), r.c_str());
    }
 
    return 0;
}
