using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Task05
{
    class Program
    {
        static void Main(string[] args)
        {
            List<int> toAdd = new List<int>();
            List<int> toDelete = new List<int>(); 
            List<int> toFind = new List<int>();
            
            for (var i = 0; i < 100; i++)
            {
                var rng = new Random();
                toAdd.Add(rng.Next(0, 100));
            }
            
            for (var i = 0; i < 50; i++)
            {
                var rng = new Random();
                toDelete.Add(rng.Next(0, 100));
            }
            
            for (var i = 0; i < 100; i++)
            {
                var rng = new Random();
                toFind.Add(rng.Next(0, 200));
            }
            
            var tree = new CoarseGrainedBST();
            var tasks = new List<Task>();
            foreach (var value in toAdd)
            {
                tasks.Add(Task.Run(() => tree.add(value)));
            }
            
            foreach (var value in toDelete)
            {
                tasks.Add(Task.Run(() => tree.delete(value)));
            }
            
            foreach (var value in toFind)
            {
                tasks.Add(Task.Run(() => tree.find(value)));
            }

            Task.WaitAll(tasks.ToArray());
            tree.print();
            Console.WriteLine("");
            var tree1 = new FineGrainedBST();
            tasks = new List<Task>();
            foreach (var value in toAdd)
            {
                tasks.Add(Task.Run(() => tree1.add(value)));
            }
            
            foreach (var value in toDelete)
            {
                tasks.Add(Task.Run(() => tree1.delete(value)));
            }
            
            foreach (var value in toFind)
            {
                tasks.Add(Task.Run(() => tree1.find(value)));
            }

            Task.WaitAll(tasks.ToArray());
            tree1.print();
        }
    }
}