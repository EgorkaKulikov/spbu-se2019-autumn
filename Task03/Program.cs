using System;
using System.Collections.Generic;
using System.Threading;

namespace ConsoleApp1
{
    class Program
    {
        static void Main(string[] args)
        {
            List<Producer> prod_list = new List<Producer>();
            List<Consumer> cons_list = new List<Consumer>();
            for (int i = 0; i < Data.num_prods_cons; i++)
            {
                Producer prod = new Producer(i.ToString());
                Consumer cons = new Consumer(i.ToString());
                prod_list.Add(prod);
                cons_list.Add(cons);
            }
            Console.ReadKey();
            Data.end_prog = true;
            //waiting for all producers to end
            while (Data.prod_ended != Data.num_prods_cons)
            {
                Thread.Sleep(100);
            }

            //imitate producer
            for (int i = 0; i < Data.num_prods_cons; i++)
            {
                Data.mutex.WaitOne();//cause consumers may be still active
                Data.buff.Add(Data.rnd.Next(0, 1000));
            }
            for (int i = 0; i < Data.num_prods_cons; i++)
            {
                Data.full.Release();
            }
            //waiting for all consumers to end
            while (Data.cons_ended != Data.num_prods_cons)
            {
                Thread.Sleep(100);
            }
            Data.buff.Clear();
        }
    }
}