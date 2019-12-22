using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using NUnit.Framework;

namespace Task05
{
    public class OptimisticTests
    {
        [Test]
        public void MutexTest()
        {
            List<Tuple<int, int>> insertList = new List<Tuple<int, int>>();
            List<int> findList = new List<int>();
            var tree = new OptimisticBinaryTree();
            var tasks = new List<Task>();

            for (int i = 0; i < 100; i++)
            {
                var random = new Random();
                insertList.Add(new Tuple<int, int>(random.Next(0, 100),random.Next(0, 100)));
                findList.Add(random.Next(0, 100));
            }

            foreach (var node in insertList)
                tasks.Add(Task.Run(() => tree.insert(node.Item1, node.Item2)));

            foreach (var key in findList)
                tasks.Add(Task.Run(() => tree.find(key)));

            Task.WaitAll(tasks.ToArray());
            Assert.True(tree.mutexesRealesed(tree.root));
        }

        [Test]
        public void IsBinaryTreeTest()
        {
            List<Tuple<int, int>> insertList = new List<Tuple<int, int>>();
            List<int> findList = new List<int>();
            var tree = new OptimisticBinaryTree();
            var tasks = new List<Task>();

            for (int i = 0; i < 100; i++)
            {
                var random = new Random();
                insertList.Add(new Tuple<int, int>(random.Next(0, 100),random.Next(0, 100)));
                findList.Add(random.Next(0, 100));
            }

            foreach (var node in insertList)
                tasks.Add(Task.Run(() => tree.insert(node.Item1, node.Item2)));

            foreach (var key in findList)
                tasks.Add(Task.Run(() => tree.find(key)));

            Task.WaitAll(tasks.ToArray());
            Assert.True(tree.isBinaryTree(tree.root));
        }
        
        [Test]
        public void findInsertTest()
        {
            var tree = new OptimisticBinaryTree();

            Thread[] insertThreads = new Thread[10];
            Thread[] findThreads = new Thread[20];
            for (int i = 0; i < 10; i++)
            {
                int i_copy = i;
                insertThreads[i_copy] = new Thread(() => {
                    tree.insert(i_copy, i_copy);
                });
                insertThreads[i_copy].Start();
            }
            
            foreach (var thread in insertThreads)
                thread.Join();
            
            for (int i = 0; i < 20; i++)
            {
                int i_copy = i;
                findThreads[i_copy] = new Thread(() => {
                    if (i_copy < 10)
                        Assert.True(tree.find(i_copy) == i_copy);
                    else
                        Assert.True(tree.find(i_copy) == null);

                });
                findThreads[i_copy].Start();
            }
            foreach (var thread in findThreads)
                thread.Join();
        }
    }
}