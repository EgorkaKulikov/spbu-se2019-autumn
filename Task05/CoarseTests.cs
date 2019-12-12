using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Task05;
using NUnit.Framework;
using NUnit.Framework.Internal;

namespace TestProject2
{
    [TestFixture]
    public class CoarseTests
    {
        
        [Test]
        public void CoarseCheckLeftPropertyKey()
        {
            var tree = new CoarseBinaryTree<int, int>();
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
        public void CoarseCheckRightPropertyKey()
        {
            var tree = new CoarseBinaryTree<int, int>();
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
        public void CoarseFindInsertTest()
        {
            var tree = new CoarseBinaryTree<int, int>();
            var flag = true;
            Parallel.For(0, 1000, i =>
            {
                tree.Insert(i, i);
            });
            for (var i = 0; i < 1000; i++)
            {
                if (tree.Find(i) != null) continue;
                flag = false;
                break;
            }
            Assert.True(flag);
        }

        [Test]
        public void CoarseDeleteTest()
        {
            var tree = new CoarseBinaryTree<int, int>();
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