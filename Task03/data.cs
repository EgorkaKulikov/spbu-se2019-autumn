using System;
using System.Threading;
using System.Collections.Generic;

namespace ConsoleApp1
{
    static class Data
    {
        public static List<int> buff = new List<int>();
        public static Semaphore may_change = new Semaphore(1, 1);
        public static Semaphore full = new Semaphore(0, int.MaxValue);
        public static Random rnd = new Random();
        public static int sleep_time = 1000;
        public static bool end_prog = false;
    }
}