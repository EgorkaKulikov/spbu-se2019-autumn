using System;
using System.Collections.Generic;

namespace Task05
{
    static class Program
    {
        public static void Main(string[] args)
        {
            int size = 5;

            BST<int> bst = new BST<int>();

            //generate keys
            Random rnd = new Random();
            int maxBound = size * size;

            var set = new HashSet<int>();
            var keys = new List<int>();

            while (keys.Count < size)
            {
                int key;
                do
                {
                    key = rnd.Next(maxBound);
                } 
                while (set.Contains(key));

                set.Add(key);
                keys.Add(key);
            }

            foreach (var key in keys)
            {
                bst.Insert(key, rnd.Next(1, maxBound));
            }
            
            bst.Show();
            Console.WriteLine();
        }
    }
}