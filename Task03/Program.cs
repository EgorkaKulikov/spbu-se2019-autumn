using System;
using System.Collections.Generic;

namespace ConsoleApp1
{
    class Program
    {
        static void Main(string[] args)
        {
            int num_prods_cons = 10;
            List<Producer> prod_list = new List<Producer>();
            List<Consumer> cons_list = new List<Consumer>();
            for (int i = 0; i < num_prods_cons; i++)
            {
                Producer prod = new Producer(i.ToString());
                Consumer cons = new Consumer(i.ToString());
                prod_list.Add(prod);
                cons_list.Add(cons);
            }
            Console.ReadKey();
            Data.end_prog = true;
        }
    }
}
