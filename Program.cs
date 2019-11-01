using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Threading;

namespace Task03
{
    internal class Program
    {
        private static readonly int countOfProdusers = 8;
        private static readonly int countOfConsumers = 8;
        public static void Main(string[] args)
        {
            var producers = new List<Producer<int>>();
            var consumers = new List<Consumer<int>>();
            for (var i = 0; i < countOfProdusers; i++)
            {
                producers.Add(new Producer<int>(i.ToString()));
            }

            for (var j = 0; j < countOfConsumers; j++)
            {
                consumers.Add(new Consumer<int>(j.ToString()));
            }
            
            Console.ReadKey();
            
            foreach (var p in producers)
            {
                p.SetCancel();
            }

            foreach (var c in consumers)
            {
                c.setCansel();
            }
        }

    }
}