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
	void testSize100();
	void testSize500();
	void testSize1000();
	void testSize1500();
	void testSize2000();
	void testSize2500();
	void testSize3000();
	void testSize3500();
	void runAllTests();
	void getAllTimes();
	void getTimesWithoutErr();
private:
    bool passedTest(int size);
    void getTime(int size, string method);
    vector<int> _sizes = {100, 500, 1000, 1500, 2000
                         , 2500, 3000, 3500};
    vector<string> _methods = {"parallel", "sequential", "gsl"};
};
