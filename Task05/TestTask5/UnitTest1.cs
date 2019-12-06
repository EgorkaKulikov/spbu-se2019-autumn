using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Threading;

using Task05;
using System.Threading.Tasks;

namespace TestTask5
{
    [TestClass]
    public class UnitTestCoarse
    {
        [TestMethod]
        public void TestAdd()
        {
            CoarseAVL tree = new CoarseAVL();
            Parallel.For(0, 4, (i, state) =>
            {
                tree.Add(i);
            });
            Assert.AreEqual("0 1 2 3 ", tree.DisplayTree());
        }
        [TestMethod]
        public void TestDelete()
        {
            CoarseAVL tree = new CoarseAVL();
            Parallel.For(0, 11, (i, state) =>
            {
                tree.Add(i);
            });
            Parallel.For(1, 4, (i, state) =>
            {
                tree.Delete(i * 3);
            });
            Assert.AreEqual("0 1 2 4 5 7 8 10 ", tree.DisplayTree());
        }
        [TestMethod]
        public void TestFind()
        {
            CoarseAVL tree = new CoarseAVL();
            bool[] ans = new bool[5];
            Parallel.For(0, 4, (i, state) =>
            {
                tree.Add(i);
            });
            Parallel.For(0, 5, (i, state) =>
            {
                ans[i] = tree.Find(i);
            });
            Assert.AreEqual(true, ans[2]);
            Assert.AreEqual(false, ans[4]);
        }
    }
    [TestClass]
    public class UnitTestLazy
    {
        [TestMethod]
        public void TestAdd()
        {
            LazyAVL tree = new LazyAVL();
            Parallel.For(0, 4, (i, state) =>
            {
                tree.Add(i);
            });
            Assert.AreEqual("0 1 2 3 ", tree.DisplayTree());
        }
        [TestMethod]
        public void TestDelete()
        {
            LazyAVL tree = new LazyAVL();
            Parallel.For(0, 11, (i, state) =>
            {
                tree.Add(i);
            });
            Parallel.For(1, 4, (i, state) =>
            {
                tree.Delete(i * 3);
            });
            Assert.AreEqual("0 1 2 4 5 7 8 10 ", tree.DisplayTree());
        }
        public void TestRealDelete()
        {
            LazyAVL tree = new LazyAVL();
            Parallel.For(0, 11, (i, state) =>
            {
                tree.Add(i);
            });
            Parallel.For(1, 4, (i, state) =>
            {
                tree.Delete(i * 3);
            });
            tree.RealDelete();
            Assert.AreEqual("0 1 2 4 5 7 8 10 ", tree.DisplayTree());
        }
        [TestMethod]
        public void TestFind()
        {
            LazyAVL tree = new LazyAVL();
            bool[] ans = new bool[5];
            Parallel.For(0, 4, (i, state) =>
            {
                tree.Add(i);
            });
            Parallel.For(0, 5, (i, state) =>
            {
                ans[i] = tree.Find(i);
            });
            Assert.AreEqual(true, ans[2]);
            Assert.AreEqual(false, ans[4]);
        }
    }
}
