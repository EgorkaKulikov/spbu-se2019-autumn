using NUnit.Framework;
using Parallel_Trees;

namespace Tests_Trees
{
    [TestFixture]
    public class TestsParallel
    {
        [Test]
        public static void InsertFindCoarse()
        {
            TestsParallelBinaryTree.InsertFind_EmptyTree(new CoarseGrainedBinaryTree<int, int>());
        }

        [Test]
        public static void InsertFindFine()
        {
            TestsParallelBinaryTree.InsertFind_EmptyTree(new FineGrainedBinaryTree<int, int>());
        }

        [Test]
        public static void FindRemoveCoarse_FilledTree()
        {
            TestsParallelBinaryTree.FindRemove_FilledTree(new CoarseGrainedBinaryTree<int, int>());
        }
        
        [Test]
        public static void FindRemoveFine_FilledTree()
        {
            TestsParallelBinaryTree.FindRemove_FilledTree(new FineGrainedBinaryTree<int, int>());
        }

        [Test]
        public static void InsertFindRemoveCoarse()
        {
            TestsParallelBinaryTree.InsertFindRemove(new CoarseGrainedBinaryTree<int, int>());
        }
        
        [Test]
        public static void InsertFindRemoveFine()
        {
            TestsParallelBinaryTree.InsertFindRemove(new FineGrainedBinaryTree<int, int>());
        }
    }
}