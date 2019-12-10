using System;
using System.Threading;
using System.Collections.Generic;

namespace ConsoleApp1
{
    static class Data
    {
        public static List<int> buff = new List<int>();
        public static Mutex mutex = new Mutex();
        public static Semaphore full = new Semaphore(0, int.MaxValue);
        public static Random rnd = new Random();
        public static int sleep_time = 1000;
        public static int num_prods_cons = 10;
        public static bool end_prog = false;
        public static int prod_ended = 0;
        public static int cons_ended = 0;
    }
}