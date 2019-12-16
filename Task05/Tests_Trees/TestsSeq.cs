using NUnit.Framework;
using Parallel_Trees;

namespace Tests_Trees
{
    [TestFixture]
    public class TestsSeq
    {
        private static T TreeWithRoot<T>() where T : BinaryTree<int, int>, new ()
        {
            var tree = new T();

            tree.Insert(2, 2);

            return tree;
        }
        
        [Test]
        public void InsertRootCoarse()
        {
            TestsSeqBinaryTree.Insert_Root(new CoarseGrainedBinaryTree<int, int>());
        }
        
        [Test]
        public void InsertRootFine()
        {
            TestsSeqBinaryTree.Insert_Root(new FineGrainedBinaryTree<int, int>());
        }

        [Test]
        public void InsertLeafCoarse()
        {
            TestsSeqBinaryTree.Insert_Leaf(new CoarseGrainedBinaryTree<int, int>());
        }
        
        [Test]
        public void InsertLeafFine()
        {
            TestsSeqBinaryTree.Insert_Leaf(new FineGrainedBinaryTree<int, int>());
        }

        [Test]
        public void InsertExistingCoarse()
        {
            TestsSeqBinaryTree.Insert_Existing(new CoarseGrainedBinaryTree<int, int>());
        } 
        
        [Test]
        public void InsertExistingFine()
        {
            TestsSeqBinaryTree.Insert_Existing(new FineGrainedBinaryTree<int, int>());
        }

        [Test]
        public void FindNonExistingCoarse()
        {
            TestsSeqBinaryTree
                .Find_NonExisting_ReturnNull(new CoarseGrainedBinaryTree<int, int>());
        }
        
        [Test]
        public void FindNonExistingFine()
        {
            TestsSeqBinaryTree
                .Find_NonExisting_ReturnNull(new FineGrainedBinaryTree<int, int>());
        }

        [Test]
        public void FindRootCoarse()
        {
            TestsSeqBinaryTree
                .Find_Root_ReturnRealRoot(new CoarseGrainedBinaryTree<int, int>());
        }
        
        [Test]
        public void FindRootFine()
        {
            TestsSeqBinaryTree
                .Find_Root_ReturnRealRoot(new FineGrainedBinaryTree<int, int>());
        }

        [Test]
        public void RemoveCoarse_RootLeaf_ReturnTrue()
        {
            TestsSeqBinaryTree
                .Remove_RootLeaf_ReturnTrue(new CoarseGrainedBinaryTree<int, int>());
        }
        
        [Test]
        public void RemoveFine_RootLeaf_ReturnTrue()
        {
            TestsSeqBinaryTree
                .Remove_RootLeaf_ReturnTrue(new FineGrainedBinaryTree<int, int>());
        }

        [Test]
        public void RemoveCoarse_NonExisting()
        {
            TestsSeqBinaryTree
                .Remove_NonExisting_ReturnFalse(new CoarseGrainedBinaryTree<int, int>());
        }
        
        [Test]
        public void RemoveFine_NonExisting()
        {
            TestsSeqBinaryTree
                .Remove_NonExisting_ReturnFalse(new FineGrainedBinaryTree<int, int>());
        }

        [Test]
        public void RemoveCoarse_RightLeaf()
        {
            TestsSeqBinaryTree.Remove_RightLeaf_ReturnTrue(new CoarseGrainedBinaryTree<int, int>());
        }
        
        [Test]
        public void RemoveFine_RightLeaf()
        {
            TestsSeqBinaryTree.Remove_RightLeaf_ReturnTrue(new FineGrainedBinaryTree<int, int>());
        }
        
        [Test]
        public void RemoveCoarse_LeftLeaf()
        {
            TestsSeqBinaryTree.Remove_LeftLeaf_ReturnTrue(new CoarseGrainedBinaryTree<int, int>());
        }
        
        [Test]
        public void RemoveFine_LeftLeaf()
        {
            TestsSeqBinaryTree.Remove_LeftLeaf_ReturnTrue(new FineGrainedBinaryTree<int, int>());
        }

        [Test]
        public void RemoveCoarse_NodeWithRightSon()
        {
            TestsSeqBinaryTree
                .Remove_NodeWithRightSon(new CoarseGrainedBinaryTree<int, int>());
        }
        
        [Test]
        public void RemoveFine_NodeWithRightSon()
        {
            TestsSeqBinaryTree
                .Remove_NodeWithRightSon(new FineGrainedBinaryTree<int, int>());
        }

        [Test]
        public void RemoveCoarse_NodeWithTwoSonsNoNeededGrandsons()
        {
            TestsSeqBinaryTree
                .Remove_NodeWithTwoSonsNoNeededGrandsons(new CoarseGrainedBinaryTree<int, int>());
        }
        
        [Test]
        public void RemoveFine_NodeWithTwoSonsNoNeededGrandsons()
        {
            TestsSeqBinaryTree
                .Remove_NodeWithTwoSonsNoNeededGrandsons(new FineGrainedBinaryTree<int, int>());
        }

        [Test]
        public void RemoveCoarse_RootWithTwoSonsNoNeededGrandsons()
        {
            TestsSeqBinaryTree
                .Remove_RootWithTwoSonsNoNeededGrandsons(new CoarseGrainedBinaryTree<int, int>());
        }
        
        [Test]
        public void RemoveFine_RootWithTwoSonsNoNeededGrandsons()
        {
            TestsSeqBinaryTree
                .Remove_RootWithTwoSonsNoNeededGrandsons(new FineGrainedBinaryTree<int, int>());
        }

        [Test]
        public void RemoveCoarse_RootWithRightSonDeepestGrandson()
        {
            TestsSeqBinaryTree
                .Remove_RootWithRightSonDeepestGrandson(new CoarseGrainedBinaryTree<int, int>());
        }
        
        [Test]
        public void RemoveFine_RootWithRightSonDeepestGrandson()
        {
            TestsSeqBinaryTree
                .Remove_RootWithRightSonDeepestGrandson(new FineGrainedBinaryTree<int, int>());
        }

        [Test]
        public void RemoveCoarse_RootWithLeftSonDeepestGrandson()
        {
            TestsSeqBinaryTree
                .Remove_RootWithLeftSonDeepestGrandson(new CoarseGrainedBinaryTree<int, int>());
        }
        
        [Test]
        public void RemoveFine_RootWithLeftSonDeepestGrandson()
        {
            TestsSeqBinaryTree
                .Remove_RootWithLeftSonDeepestGrandson(new FineGrainedBinaryTree<int, int>());
        }

        [Test]
        public static void RemoveCoarse_NodeWithLeftSon()
        {
            TestsSeqBinaryTree.Remove_NodeWithLeftSon(new CoarseGrainedBinaryTree<int, int>());
        }
        
        [Test]
        public static void RemoveFine_NodeWithLeftSon()
        {
            TestsSeqBinaryTree.Remove_NodeWithLeftSon(new FineGrainedBinaryTree<int, int>());
        }

        [Test]
        public static void RemoveCoarse_RootLeftSonLeftGrandSon()
        {
            TestsSeqBinaryTree.Remove_RootLeftSonLeftGrandSon(new CoarseGrainedBinaryTree<int, int>());
        }
        
        [Test]
        public static void RemoveFine_RootLeftSonLeftGrandSon()
        {
            TestsSeqBinaryTree.Remove_RootLeftSonLeftGrandSon(new FineGrainedBinaryTree<int, int>());
        }

        [Test]
        public static void RemoveCoarse_RootRightSonRightGrandSon()
        {
            TestsSeqBinaryTree.Remove_RootRightSonRightGrandSon(new CoarseGrainedBinaryTree<int, int>());
        }
        
        [Test]
        public static void RemoveFine_RootRightSonRightGrandSon()
        {
            TestsSeqBinaryTree.Remove_RootRightSonRightGrandSon(new FineGrainedBinaryTree<int, int>());
        }
    }
}