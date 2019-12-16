using NUnit.Framework;
using Parallel_Trees;

namespace Tests_Trees
{
    internal class TestsSeqBinaryTree
    {
        private static void CheckRelations(CoarseGrainedBinaryTree<int, int>.Node parent
            , CoarseGrainedBinaryTree<int, int>.Node son, bool isLeftSon)
        {
            if (parent != null) Assert.AreEqual(son, isLeftSon ? parent.LeftSon : parent.RightSon);
            Assert.AreEqual(parent, son.Parent);
        }
        
        internal static void Insert_Root(BinaryTree<int, int> tree)
        {
            var root = tree.Insert(5, 6);
            
            Assert.AreEqual(tree.Root, root);
            Assert.AreEqual(null, root.LeftSon);
            Assert.AreEqual(null, root.RightSon);
            Assert.AreEqual(null, root.Parent);
        }
        
        internal static void Insert_Leaf(BinaryTree<int, int> tree)
        {
            var parent = tree.Insert(11, 11);
            var newLeaf = tree.Insert(10, 10);
            
            Assert.AreEqual(newLeaf, parent.LeftSon);
            Assert.AreEqual(parent, newLeaf.Parent);
            Assert.AreEqual(null, newLeaf.LeftSon);
            Assert.AreEqual(null, newLeaf.RightSon);
        }
        
        internal static void Insert_Existing(BinaryTree<int, int> tree)
        {
            tree.Insert(9, 9);
            
            tree.Insert(9, 10);

            Assert.AreEqual(10, tree.Find(9));
        }
        
        internal static void Find_NonExisting_ReturnNull(BinaryTree<int, int> tree)
        {
            var res = tree.Find(0);
            
            Assert.AreEqual(null, res);
        }
        
        internal static void Find_Root_ReturnRealRoot(BinaryTree<int, int> tree)
        {
            tree.Insert(20, 20);
            
            Assert.AreEqual(20, tree.Find(20));
        }

        internal static void Remove_RootLeaf_ReturnTrue(BinaryTree<int, int> tree)
        {
            tree.Insert(20, 20);
            
            var res = tree.Remove(20);
            
            Assert.IsTrue(res);
            Assert.AreEqual(null, tree.Root);
        }

        internal static void Remove_NonExisting_ReturnFalse(BinaryTree<int, int> tree)
        {
            var res = tree.Remove(0);
            
            Assert.IsFalse(res);
        }
        
        
        internal static void Remove_RightLeaf_ReturnTrue(BinaryTree<int, int> tree)
        {
            var parent = tree.Insert(2, 2);
            var leaf = tree.Insert(7, 7);

            var res = tree.Remove(7);

            Assert.IsTrue(res);
            Assert.IsNull(parent.RightSon);
            Assert.IsNull(tree.Find(7));
        }
        
        internal static void Remove_LeftLeaf_ReturnTrue(BinaryTree<int, int> tree)
        {
            var parent = tree.Insert(2, 2);
            var leaf = tree.Insert(1, 1);

            var res = tree.Remove(1);

            Assert.IsTrue(res);
            Assert.IsNull(parent.LeftSon);
            Assert.IsNull(tree.Find(1));
        }
        
        internal static void Remove_NodeWithTwoSonsNoNeededGrandsons(BinaryTree<int, int> tree)
        {
            var parent = tree.Insert(20, 20);
            tree.Insert(12, 12);
            var rightSon = tree.Insert(18, 18);
            var leftSon = tree.Insert(4, 4);

            var res = tree.Remove(12);

            Assert.IsTrue(res);
            Assert.AreEqual(20, parent.Key);
            Assert.AreEqual(18, rightSon.Key);
            Assert.AreEqual(4, leftSon.Key);
            
            CheckRelations(parent, leftSon, true);
            CheckRelations(leftSon, rightSon, false);
        }
        
        internal static void Remove_RootWithTwoSonsNoNeededGrandsons(BinaryTree<int, int> tree)
        {
            tree.Insert(12, 12);
            var rightSon = tree.Insert(18, 18);
            var leftSon = tree.Insert(4, 4);

            var res = tree.Remove(12);

            Assert.IsTrue(res);
            Assert.AreEqual(4, leftSon.Key);
            Assert.AreEqual(18, rightSon.Key);
            
            CheckRelations(leftSon, rightSon, false);
        }

        internal static void Remove_NodeWithLeftSon(BinaryTree<int, int> tree)
        {
            var parent = tree.Insert(20, 20);
            tree.Insert(12, 12);
            var rightSon = tree.Insert(16, 16);
            var grandson = tree.Insert(18, 18);

            var res = tree.Remove(12);

            Assert.IsTrue(res);
            Assert.AreEqual(20, parent.Key);
            Assert.AreEqual(16, rightSon.Key);
            Assert.AreEqual(18, grandson.Key);
            
            CheckRelations(parent, rightSon, true);
            CheckRelations(rightSon, grandson, false);
        }
        
        internal static void Remove_NodeWithRightSon(BinaryTree<int, int> tree)
        {
            var parent = tree.Insert(20, 20);
            tree.Insert(30, 30);
            var leftSon = tree.Insert(26, 26);
            var grandson = tree.Insert(22, 22);
            
            var res = tree.Remove(30);
            
            Assert.IsTrue(res);
            
            CheckRelations(parent, leftSon, false);
            CheckRelations(leftSon, grandson, true);
        }
        
        internal static void Remove_RootWithRightSonDeepestGrandson(BinaryTree<int, int> tree)
        {
            var root = tree.Insert(20, 20);
            var rightSon = tree.Insert(30, 30);
            tree.Insert(23, 23);
            var grandsonRightSubTree = tree.Insert(24, 24);

            var res = tree.Remove(20);
            
            Assert.IsTrue(res);
            Assert.AreEqual(23, root.Key);
            Assert.AreEqual(30, rightSon.Key);
            Assert.AreEqual(24, grandsonRightSubTree.Key);
            
            CheckRelations(rightSon, grandsonRightSubTree, true);
            CheckRelations(root, rightSon, false);
        }
        
        internal static void Remove_RootWithLeftSonDeepestGrandson(BinaryTree<int, int> tree)
        {
            var root = tree.Insert(20, 20);
            var leftSon = tree.Insert(10, 10);
            tree.Insert(18, 18);
            var grandsonLeftSubTree = tree.Insert(16, 16);

            var res = tree.Remove(20);
            
            Assert.IsTrue(res);
            Assert.AreEqual(10, leftSon.Key);
            Assert.AreEqual(18, root.Key);
            Assert.AreEqual(16, grandsonLeftSubTree.Key);
            
            CheckRelations(leftSon, grandsonLeftSubTree, false);
            CheckRelations(root, leftSon, true);
        }

        internal static void Remove_RootLeftSonLeftGrandSon(BinaryTree<int, int> tree)
        {
            tree.Insert(20, 20);
            var leftSon = tree.Insert(10, 10);
            var grandsonLeftSubTree = tree.Insert(5, 5);

            var res = tree.Remove(20);
            
            Assert.IsTrue(res);
            Assert.AreEqual(10, leftSon.Key);
            Assert.AreEqual(5, grandsonLeftSubTree.Key);
            
            Assert.AreEqual(tree.Root, leftSon);
            CheckRelations(leftSon, grandsonLeftSubTree, true);
        }
        
        internal static void Remove_RootRightSonRightGrandSon(BinaryTree<int, int> tree)
        {
            tree.Insert(20, 20);
            var rightSon = tree.Insert(30, 30);
            var grandsonRightSubTree = tree.Insert(40, 40);

            var res = tree.Remove(20);
            
            Assert.IsTrue(res);
            Assert.AreEqual(30, rightSon.Key);
            Assert.AreEqual(40, grandsonRightSubTree.Key);
            
            Assert.AreEqual(tree.Root, rightSon);
            CheckRelations(rightSon, grandsonRightSubTree, false);
        }
    }
}