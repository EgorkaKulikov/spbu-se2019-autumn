#pragma once
#include "test.h"
#include "gauss.h"
#include <string>
#include <vector>

using std::string;
using std::vector;

void setPrecision(double &err, double &res);

class GaussTest : public Test {
public:
	void testSize10();
	void testSize100();
	void testSize500();
	void testSize1000();
	void testSize2000();
	void testSize3000();
	void testSize4000();
	void testSize5000();
	void testSize10000();
	void runAllTests();
	void getAllTimes();
private:
    bool passedTest(int size);
    void getTime(int size, string method);
    vector<int> _sizes = {100, 500, 1000, 2000, 3000
                         , 4000, 5000, 10000};
    vector<string> _methods = {"parallel", "sequential", "gsl"};
};

