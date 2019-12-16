using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using NUnit.Framework;
using Task05;

namespace TestProject2
{
    [TestFixture]
    public class FineTests
    {
        private static Random rng = new Random();
        private static void Shuffle<T>(IList<T> list)  
        {  
            var n = list.Count;  
            while (n > 1) {  
                n--;  
                var k = rng.Next(n + 1);  
                var value = list[k];  
                list[k] = list[n];  
                list[n] = value;  
            }  
        }
        [Test]
        public void FineCheckLeftPropertyKey()
        {
            var tree = new FineBinaryTree<int, int>();
            Parallel.For(0, 1000, i =>
            {
                tree.Insert(i, i);
            });
            var cur = tree.Root;
            var actual = true;

            while (cur.Left != null)
            {
                actual = cur.Value > cur.Left.Value;
                if (!actual) break;
                cur = cur.Left;
            }
            Assert.True(actual);
        }
        
        [Test]
        public void FineCheckRightPropertyKey()
        {
            var tree = new FineBinaryTree<int, int>();
            Parallel.For(0, 1000, i =>
            {
                tree.Insert(i, i);
            });
            var cur = tree.Root;
            var actual = true;

            while (cur.Right != null)
            {
                actual = cur.Value < cur.Right.Value;
                if (!actual) break;
                cur = cur.Right;
            }
            Assert.True(actual);
        }
        
        [Test]
        public void FineFindInsertTest()
        {
            var tree = new FineBinaryTree<int, int>();
            var list = new List<int>();
            for (var i = 0; i < 10000; i++)
            {
                list.Add(i);
            }
            Shuffle(list);
            Parallel.ForEach(list, i => tree.Insert(i, i)); 
            var flag = true;
            Parallel.For(0, 1000, i =>
            {
                if (tree.Find(i) == null)
                {
                    flag = false;
                }
            });
            Parallel.For(0, 1000, i =>
            {
                if (tree.Find(i) == null)
                {
                    flag = false;
                }
            });
            
            Assert.True(flag);
        }

        [Test]
        public void FineDeleteTest()
        {
            var tree = new FineBinaryTree<int, int>();
            Parallel.For(0, 1000, i =>
            {
                tree.Insert(i, i);
            });
            Parallel.For(0, 1000, i =>
            {
                tree.Delete(i);
            });
            Assert.IsNull(tree.Root);
        }

    }
}