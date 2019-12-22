using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;

namespace Task05
{
    internal class Program
    {
        public static void Main(string[] args)
        {
            List<Tuple<int, int>> insertList = new List<Tuple<int, int>>();
            List<int> findList = new List<int>();
            var tree = new FineBinaryTree();
            var tasks = new List<Task>();

            for (int i = 0; i < 1000000; i++)
            {
                var random = new Random();
                insertList.Add(new Tuple<int, int>(random.Next(0, 1000000),random.Next(0, 1000000)));
                findList.Add(random.Next(0, 1000000));
            }
            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();

            foreach (var node in insertList)
                tasks.Add(Task.Run(() => tree.insert(node.Item1, node.Item2)));

            foreach (var key in findList)
                tasks.Add(Task.Run(() => tree.find(key)));

            Task.WaitAll(tasks.ToArray());
            stopWatch.Stop();
            Console.WriteLine("RunTime " + Convert.ToString(stopWatch.ElapsedMilliseconds));
            
        }
    }
}