using Microsoft.VisualStudio.TestTools.UnitTesting;
using Task05;
using System.Threading;

namespace Task05Tests
{
    [TestClass]
    public class FineSynchronizedBinarySearchTreeTests
    {
        private FineSynchronizedBinarySearchTree<int, int> SequentialCreateFSBST(int[] requests)
        {
            var tree = new FineSynchronizedBinarySearchTree<int, int>();
            var k = 0;
            foreach (var request in requests)
            {
                if (request < 0)
                    tree[-request] = null;
                else
                    tree[request] = ++k;
            }
            return tree;
        }

        private FineSynchronizedBinarySearchTree<int, int> ParallelCreateFSBST(int[] requests)
        {
            var tree = new FineSynchronizedBinarySearchTree<int, int>();
            var amountThreads = 4;
            var threads = new Thread[amountThreads];
            var k = 0;
            for (int i = 0; i < amountThreads; i++)
            {
                var ind = i;
                threads[ind] = new Thread(() =>
                {
                    for (int j = ind; j < requests.Length; j += amountThreads)
                    {
                        if (requests[j] < 0)
                            tree[-requests[j]] = null;
                        else
                            tree[requests[j]] = ++k;
                    }
                });
                threads[ind].Start();
            }
            for (int ind = 0; ind < amountThreads; ind++)
                threads[ind].Join();
            return tree;
        }

        [DataTestMethod]
        [DataRow(new int[] { })] // 1
        [DataRow(new int[] { 3 })] // 2
        [DataRow(new int[] { 3, 9 })] // 3
        [DataRow(new int[] { 3, 9, 5 })] // 4
        [DataRow(new int[] { 3, 9, 5, 4 })] // 5
        [DataRow(new int[] { 3, 9, 5, 4, 7 })] // 6
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1 })] // 7
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6 })] // 8
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6, 10 })] // 9
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6, 10, 8 })] // 10
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6, 10, 8, 2 })] // 11
        [DataRow(new int[] { 7 })] // 12
        [DataRow(new int[] { 7, 2 })] // 13
        [DataRow(new int[] { 7, 2, 9 })] // 14
        [DataRow(new int[] { 7, 2, 9, 1 })] // 15
        [DataRow(new int[] { 7, 2, 9, 1, -2 })] // 16
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4 })] // 17
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9 })] // 18
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9 })] // 19
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10 })] // 20
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8 })] // 21
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7 })] // 22
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2 })] // 23
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2 })] // 24
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10 })] // 25
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3 })] // 26
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2 })] // 27
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1 })] // 28
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13 })] // 29
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9 })] // 30
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9, 7 })] // 31
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9, 7, -8 })] // 32
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9, 7, -8, -2 })] // 33
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9, 7, -8, -2, -10 })] // 34
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9, 7, -8, -2, -10, -3 })] // 35
        [DataRow(new int[] { 5 })] // 36
        [DataRow(new int[] { 5, -5 })] // 37
        [DataRow(new int[] { 5, -5, 8 })] // 38
        [DataRow(new int[] { 5, -5, 8, 8 })] // 39
        [DataRow(new int[] { 5, -5, 8, 8, -8 })] // 40
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8 })] // 41
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4 })] // 42
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7 })] // 43
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4 })] // 44
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4, 4 })] // 45
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4, 4, -7 })] // 46
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4, 4, -7, 3 })] // 47
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4, 4, -7, 3, 5 })] // 48
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4, 4, -7, 3, 5, -4 })] // 49
        public void CorrectLinksFatherSonSequentialTest(int[] requests)
        {
            var tree = SequentialCreateFSBST(requests);
            Assert.IsTrue(tree.Root == null || CheckLinksFatherSon(tree.Root));

            bool CheckLinksFatherSon(BinaryNode<int, int> node) =>
                (
                    node.Left == null ||
                    (
                        node.Left.Parent == node &&
                        CheckLinksFatherSon(node.Left)
                    )
                ) &&
                (
                    node.Right == null ||
                    (
                        node.Right.Parent == node &&
                        CheckLinksFatherSon(node.Right)
                    )
                );
        }

        [DataTestMethod]
        [DataRow(new int[] { })] // 1
        [DataRow(new int[] { 3 })] // 2
        [DataRow(new int[] { 3, 9 })] // 3
        [DataRow(new int[] { 3, 9, 5 })] // 4
        [DataRow(new int[] { 3, 9, 5, 4 })] // 5
        [DataRow(new int[] { 3, 9, 5, 4, 7 })] // 6
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1 })] // 7
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6 })] // 8
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6, 10 })] // 9
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6, 10, 8 })] // 10
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6, 10, 8, 2 })] // 11
        [DataRow(new int[] { 7 })] // 12
        [DataRow(new int[] { 7, 2 })] // 13
        [DataRow(new int[] { 7, 2, 9 })] // 14
        [DataRow(new int[] { 7, 2, 9, 1 })] // 15
        [DataRow(new int[] { 7, 2, 9, 1, -2 })] // 16
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4 })] // 17
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9 })] // 18
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9 })] // 19
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10 })] // 20
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8 })] // 21
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7 })] // 22
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2 })] // 23
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2 })] // 24
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10 })] // 25
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3 })] // 26
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2 })] // 27
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1 })] // 28
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13 })] // 29
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9 })] // 30
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9, 7 })] // 31
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9, 7, -8 })] // 32
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9, 7, -8, -2 })] // 33
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9, 7, -8, -2, -10 })] // 34
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9, 7, -8, -2, -10, -3 })] // 35
        [DataRow(new int[] { 5 })] // 36
        [DataRow(new int[] { 5, -5 })] // 37
        [DataRow(new int[] { 5, -5, 8 })] // 38
        [DataRow(new int[] { 5, -5, 8, 8 })] // 39
        [DataRow(new int[] { 5, -5, 8, 8, -8 })] // 40
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8 })] // 41
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4 })] // 42
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7 })] // 43
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4 })] // 44
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4, 4 })] // 45
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4, 4, -7 })] // 46
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4, 4, -7, 3 })] // 47
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4, 4, -7, 3, 5 })] // 48
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4, 4, -7, 3, 5, -4 })] // 49
        public void CorrectKeysInSubtreesSequentialTest(int[] requests)
        {
            var tree = SequentialCreateFSBST(requests);
            Assert.IsTrue(CheckKeysInSubtrees(tree.Root, null, null));

            bool CheckKeysInSubtrees(BinaryNode<int, int> node, int? min, int? max) =>
                node == null ||
                (
                    (
                        min == null ||
                        node.Key > min
                    ) &&
                    (
                        max == null ||
                        node.Key < max
                    ) &&
                    CheckKeysInSubtrees(node.Left, min, node.Key) &&
                    CheckKeysInSubtrees(node.Right, node.Key, max)
                );
        }

        [DataTestMethod]
        [DataRow(new int[] { }, new int[] { })] // 1
        [DataRow(new int[] { 3 }, new int[] { 3 })] // 2
        [DataRow(new int[] { 3, 9 }, new int[] { 3, 9 })] // 3
        [DataRow(new int[] { 3, 9, 5 }, new int[] { 3, 5, 9 })] // 4
        [DataRow(new int[] { 3, 9, 5, 4 }, new int[] { 3, 4, 5, 9 })] // 5
        [DataRow(new int[] { 3, 9, 5, 4, 7 }, new int[] { 3, 4, 5, 7, 9 })] // 6
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1 }, new int[] { 1, 3, 4, 5, 7, 9 })] // 7
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6 }, new int[] { 1, 3, 4, 5, 6, 7, 9 })] // 8
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6, 10 }, new int[] { 1, 3, 4, 5, 6, 7, 9, 10 })] // 9
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6, 10, 8 }, new int[] { 1, 3, 4, 5, 6, 7, 8, 9, 10 })] // 10
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6, 10, 8, 2 }, new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 })] // 11
        [DataRow(new int[] { 7 }, new int[] { 7 })] // 12
        [DataRow(new int[] { 7, 2 }, new int[] { 2, 7 })] // 13
        [DataRow(new int[] { 7, 2, 9 }, new int[] { 2, 7, 9 })] // 14
        [DataRow(new int[] { 7, 2, 9, 1 }, new int[] { 1, 2, 7, 9 })] // 15
        [DataRow(new int[] { 7, 2, 9, 1, -2 }, new int[] { 1, 7, 9 })] // 16
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4 }, new int[] { 1, 4, 7, 9 })] // 17
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9 }, new int[] { 1, 4, 7 })] // 18
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9 }, new int[] { 1, 4, 7, 9 })] // 19
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10 }, new int[] { 1, 4, 7, 9, 10 })] // 20
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8 }, new int[] { 1, 4, 7, 8, 9, 10 })] // 21
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7 }, new int[] { 1, 4, 8, 9, 10 })] // 22
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2 }, new int[] { 1, 2, 4, 8, 9, 10 })] // 23
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2 }, new int[] { 1, 4, 8, 9, 10 })] // 24
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10 }, new int[] { 1, 4, 8, 9, 10 })] // 25
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3 }, new int[] { 1, 3, 4, 8, 9, 10 })] // 26
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2 }, new int[] { 1, 2, 3, 4, 8, 9, 10 })] // 27
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1 }, new int[] { 2, 3, 4, 8, 9, 10 })] // 28
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13 }, new int[] { 2, 3, 4, 8, 9, 10 })] // 29
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9 }, new int[] { 2, 3, 4, 8, 10 })] // 30
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9, 7 }, new int[] { 2, 3, 4, 7, 8, 10 })] // 31
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9, 7, -8 }, new int[] { 2, 3, 4, 7, 10 })] // 32
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9, 7, -8, -2 }, new int[] { 3, 4, 7, 10 })] // 33
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9, 7, -8, -2, -10 }, new int[] { 3, 4, 7 })] // 34
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9, 7, -8, -2, -10, -3 }, new int[] { 4, 7 })] // 35
        [DataRow(new int[] { 5 }, new int[] { 5 })] // 36
        [DataRow(new int[] { 5, -5 }, new int[] { })] // 37
        [DataRow(new int[] { 5, -5, 8 }, new int[] { 8 })] // 38
        [DataRow(new int[] { 5, -5, 8, 8 }, new int[] { 8 })] // 39
        [DataRow(new int[] { 5, -5, 8, 8, -8 }, new int[] { })] // 40
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8 }, new int[] { })] // 41
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4 }, new int[] { 4 })] // 42
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7 }, new int[] { 4, 7 })] // 43
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4 }, new int[] { 7 })] // 44
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4, 4 }, new int[] { 4, 7 })] // 45
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4, 4, -7 }, new int[] { 4 })] // 46
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4, 4, -7, 3 }, new int[] { 3, 4 })] // 47
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4, 4, -7, 3, 5 }, new int[] { 3, 4, 5 })] // 48
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4, 4, -7, 3, 5, -4 }, new int[] { 3, 5 })] // 49
        public void MatchingKeysAndAddedValuesSequentialTest(int[] requests, int[] addedValues)
        {
            var tree = SequentialCreateFSBST(requests);
            Assert.IsTrue(CheckEqualsTwoArray(tree.Keys, addedValues));

            bool CheckEqualsTwoArray(int[] a, int[] b)
            {
                if (a.Length != b.Length) return false;
                for (int i = 0; i < a.Length; i++)
                {
                    if (a[i] != b[i]) return false;
                }
                return true;
            }
        }

        [DataTestMethod]
        [DataRow(new int[] { })] // 1
        [DataRow(new int[] { 3 })] // 2
        [DataRow(new int[] { 3, 9 })] // 3
        [DataRow(new int[] { 3, 9, 5 })] // 4
        [DataRow(new int[] { 3, 9, 5, 4 })] // 5
        [DataRow(new int[] { 3, 9, 5, 4, 7 })] // 6
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1 })] // 7
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6 })] // 8
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6, 10 })] // 9
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6, 10, 8 })] // 10
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6, 10, 8, 2 })] // 11
        [DataRow(new int[] { 7 })] // 12
        [DataRow(new int[] { 7, 2 })] // 13
        [DataRow(new int[] { 7, 2, 9 })] // 14
        [DataRow(new int[] { 7, 2, 9, 10 })] // 15
        [DataRow(new int[] { 7, 2, 9, 10, 1 })] // 16
        [DataRow(new int[] { 7, 2, 9, 10, 1, 3 })] // 17
        [DataRow(new int[] { 7, 2, 9, 10, 1, 3, 4 })] // 18
        [DataRow(new int[] { 7, 2, 9, 10, 1, 3, 4, 8 })] // 19
        [DataRow(new int[] { 7, 2, 9, 10, 1, 3, 4, 8, 6 })] // 20
        [DataRow(new int[] { 7, 2, 9, 10, 1, 3, 4, 8, 6, 13 })] // 21
        [DataRow(new int[] { 7, 2, 9, 10, 1, 3, 4, 8, 6, 13, 12 })] // 22
        [DataRow(new int[] { 7, 2, 9, 10, 1, 3, 4, 8, 6, 13, 12, 5 })] // 23
        [DataRow(new int[] { 7, 2, 9, 10, 1, 3, 4, 8, 6, 13, 12, 5, 11 })] // 24
        public void CorrectValuesInSubtreesSequentialTest(int[] requests)
        {
            var tree = SequentialCreateFSBST(requests);
            Assert.IsTrue(CheckValuesInSubtrees(tree.Root, 0));

            bool CheckValuesInSubtrees(BinaryNode<int, int> node, int nodeParentValue) =>
                node == null ||
                (
                    node.Value > nodeParentValue &&
                    CheckValuesInSubtrees(node.Left, node.Value) &&
                    CheckValuesInSubtrees(node.Right, node.Value)
                );
        }

        [DataTestMethod]
        [DataRow(new int[] { })] // 1
        [DataRow(new int[] { 3 })] // 2
        [DataRow(new int[] { 3, 9 })] // 3
        [DataRow(new int[] { 3, 9, 5 })] // 4
        [DataRow(new int[] { 3, 9, 5, 4 })] // 5
        [DataRow(new int[] { 3, 9, 5, 4, 7 })] // 6
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1 })] // 7
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6 })] // 8
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6, 10 })] // 9
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6, 10, 8 })] // 10
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6, 10, 8, 2 })] // 11
        [DataRow(new int[] { 7 })] // 12
        [DataRow(new int[] { 7, 2 })] // 13
        [DataRow(new int[] { 7, 2, 9 })] // 14
        [DataRow(new int[] { 7, 2, 9, 1 })] // 15
        [DataRow(new int[] { 7, 2, 9, 1, -2 })] // 16
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4 })] // 17
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9 })] // 18
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9 })] // 19
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10 })] // 20
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8 })] // 21
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7 })] // 22
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2 })] // 23
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2 })] // 24
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10 })] // 25
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3 })] // 26
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2 })] // 27
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1 })] // 28
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13 })] // 29
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9 })] // 30
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9, 7 })] // 31
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9, 7, -8 })] // 32
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9, 7, -8, -2 })] // 33
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9, 7, -8, -2, -10 })] // 34
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9, 7, -8, -2, -10, -3 })] // 35
        [DataRow(new int[] { 5 })] // 36
        [DataRow(new int[] { 5, -5 })] // 37
        [DataRow(new int[] { 5, -5, 8 })] // 38
        [DataRow(new int[] { 5, -5, 8, 8 })] // 39
        [DataRow(new int[] { 5, -5, 8, 8, -8 })] // 40
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8 })] // 41
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4 })] // 42
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7 })] // 43
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4 })] // 44
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4, 4 })] // 45
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4, 4, -7 })] // 46
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4, 4, -7, 3 })] // 47
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4, 4, -7, 3, 5 })] // 48
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4, 4, -7, 3, 5, -4 })] // 49
        public void CorrectLinksFatherSonParallelTest(int[] requests)
        {
            var tree = ParallelCreateFSBST(requests);
            Assert.IsTrue(tree.Root == null || CheckLinksFatherSon(tree.Root));

            bool CheckLinksFatherSon(BinaryNode<int, int> node) =>
                (
                    node.Left == null ||
                    (
                        node.Left.Parent == node &&
                        CheckLinksFatherSon(node.Left)
                    )
                ) &&
                (
                    node.Right == null ||
                    (
                        node.Right.Parent == node &&
                        CheckLinksFatherSon(node.Right)
                    )
                );
        }

        [DataTestMethod]
        [DataRow(new int[] { })] // 1
        [DataRow(new int[] { 3 })] // 2
        [DataRow(new int[] { 3, 9 })] // 3
        [DataRow(new int[] { 3, 9, 5 })] // 4
        [DataRow(new int[] { 3, 9, 5, 4 })] // 5
        [DataRow(new int[] { 3, 9, 5, 4, 7 })] // 6
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1 })] // 7
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6 })] // 8
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6, 10 })] // 9
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6, 10, 8 })] // 10
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6, 10, 8, 2 })] // 11
        [DataRow(new int[] { 7 })] // 12
        [DataRow(new int[] { 7, 2 })] // 13
        [DataRow(new int[] { 7, 2, 9 })] // 14
        [DataRow(new int[] { 7, 2, 9, 1 })] // 15
        [DataRow(new int[] { 7, 2, 9, 1, -2 })] // 16
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4 })] // 17
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9 })] // 18
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9 })] // 19
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10 })] // 20
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8 })] // 21
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7 })] // 22
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2 })] // 23
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2 })] // 24
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10 })] // 25
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3 })] // 26
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2 })] // 27
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1 })] // 28
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13 })] // 29
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9 })] // 30
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9, 7 })] // 31
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9, 7, -8 })] // 32
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9, 7, -8, -2 })] // 33
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9, 7, -8, -2, -10 })] // 34
        [DataRow(new int[] { 7, 2, 9, 1, -2, 4, -9, 9, 10, 8, -7, 2, -2, 10, 3, 2, -1, -13, -9, 7, -8, -2, -10, -3 })] // 35
        [DataRow(new int[] { 5 })] // 36
        [DataRow(new int[] { 5, -5 })] // 37
        [DataRow(new int[] { 5, -5, 8 })] // 38
        [DataRow(new int[] { 5, -5, 8, 8 })] // 39
        [DataRow(new int[] { 5, -5, 8, 8, -8 })] // 40
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8 })] // 41
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4 })] // 42
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7 })] // 43
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4 })] // 44
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4, 4 })] // 45
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4, 4, -7 })] // 46
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4, 4, -7, 3 })] // 47
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4, 4, -7, 3, 5 })] // 48
        [DataRow(new int[] { 5, -5, 8, 8, -8, -8, 4, 7, -4, 4, -7, 3, 5, -4 })] // 49
        public void CorrectKeysInSubtreesParallelTest(int[] requests)
        {
            var tree = ParallelCreateFSBST(requests);
            Assert.IsTrue(CheckKeysInSubtrees(tree.Root, null, null));

            bool CheckKeysInSubtrees(BinaryNode<int, int> node, int? min, int? max) =>
                node == null ||
                (
                    (
                        min == null ||
                        node.Key > min
                    ) &&
                    (
                        max == null ||
                        node.Key < max
                    ) &&
                    CheckKeysInSubtrees(node.Left, min, node.Key) &&
                    CheckKeysInSubtrees(node.Right, node.Key, max)
                );
        }

        [DataTestMethod]
        [DataRow(new int[] { }, new int[] { })] // 1
        [DataRow(new int[] { 3 }, new int[] { 3 })] // 2
        [DataRow(new int[] { 3, 9 }, new int[] { 3, 9 })] // 3
        [DataRow(new int[] { 3, 9, 5 }, new int[] { 3, 5, 9 })] // 4
        [DataRow(new int[] { 3, 9, 5, 4 }, new int[] { 3, 4, 5, 9 })] // 5
        [DataRow(new int[] { 3, 9, 5, 4, 7 }, new int[] { 3, 4, 5, 7, 9 })] // 6
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1 }, new int[] { 1, 3, 4, 5, 7, 9 })] // 7
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6 }, new int[] { 1, 3, 4, 5, 6, 7, 9 })] // 8
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6, 10 }, new int[] { 1, 3, 4, 5, 6, 7, 9, 10 })] // 9
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6, 10, 8 }, new int[] { 1, 3, 4, 5, 6, 7, 8, 9, 10 })] // 10
        [DataRow(new int[] { 3, 9, 5, 4, 7, 1, 6, 10, 8, 2 }, new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 })] // 11
        [DataRow(new int[] { 7 }, new int[] { 7 })] // 12
        [DataRow(new int[] { 7, 2 }, new int[] { 2, 7 })] // 13
        [DataRow(new int[] { 7, 2, 9 }, new int[] { 2, 7, 9 })] // 14
        [DataRow(new int[] { 7, 2, 9, 10 }, new int[] { 2, 7, 9, 10 })] // 15
        [DataRow(new int[] { 7, 2, 9, 10, 1 }, new int[] { 1, 2, 7, 9, 10 })] // 16
        [DataRow(new int[] { 7, 2, 9, 10, 1, 3 }, new int[] { 1, 2, 3, 7, 9, 10 })] // 17
        [DataRow(new int[] { 7, 2, 9, 10, 1, 3, 4 }, new int[] { 1, 2, 3, 4, 7, 9, 10 })] // 18
        [DataRow(new int[] { 7, 2, 9, 10, 1, 3, 4, 8 }, new int[] { 1, 2, 3, 4, 7, 8, 9, 10 })] // 19
        [DataRow(new int[] { 7, 2, 9, 10, 1, 3, 4, 8, 6 }, new int[] { 1, 2, 3, 4, 6, 7, 8, 9, 10 })] // 20
        [DataRow(new int[] { 7, 2, 9, 10, 1, 3, 4, 8, 6, 13 }, new int[] { 1, 2, 3, 4, 6, 7, 8, 9, 10, 13 })] // 21
        [DataRow(new int[] { 7, 2, 9, 10, 1, 3, 4, 8, 6, 13, 12 }, new int[] { 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13 })] // 22
        [DataRow(new int[] { 7, 2, 9, 10, 1, 3, 4, 8, 6, 13, 12, 5 }, new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13 })] // 23
        [DataRow(new int[] { 7, 2, 9, 10, 1, 3, 4, 8, 6, 13, 12, 5, 11 }, new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 })] // 24
        public void MatchingKeysAndAddedValuesParallelTest(int[] requests, int[] addedValues)
        {
            var tree = ParallelCreateFSBST(requests);
            Assert.IsTrue(CheckEqualsTwoArray(tree.Keys, addedValues));

            bool CheckEqualsTwoArray(int[] a, int[] b)
            {
                if (a.Length != b.Length) return false;
                for (int i = 0; i < a.Length; i++)
                {
                    if (a[i] != b[i]) return false;
                }
                return true;
            }
        }
    }
}
