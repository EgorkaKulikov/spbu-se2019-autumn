using System.Collections.Generic;
using System.Linq;
using Parallel_Trees;

namespace Tests_Trees
{
    public static class TestUtils
    {
        internal static bool ContentEquals(BinaryTree<int, int> expected, BinaryTree<int, int> actual)
        {
            var expectedSet = new HashSet<int>();
            var actualSet = new HashSet<int>();
            ConvertTreeToHashSet(ref expectedSet, expected.Root);
            ConvertTreeToHashSet(ref actualSet, actual.Root);
            
            return expectedSet.All(actualSet.Contains) && actualSet.All(expectedSet.Contains);
        }
        
        private static HashSet<int> ConvertTreeToHashSet(ref HashSet<int> hashSet
            , BinaryTree<int, int>.Node node)
        {
            if (node == null) return hashSet;
            
            hashSet.Add(node.Key);
            ConvertTreeToHashSet(ref hashSet, node.LeftSon);
            ConvertTreeToHashSet(ref hashSet, node.RightSon);

            return hashSet;
        }
        
        internal static bool CheckRule(BinaryTree<int, int> tree)
        {
            return CheckRuleRec(tree.Root);
        }
        
        private static bool CheckRuleRec(BinaryTree<int, int>.Node currNode)
        {
            if (currNode == null) return true;

            var leftSon = currNode.LeftSon;
            var rightSon = currNode.RightSon;

            if (currNode.LeftSon != null && currNode.Key.CompareTo(leftSon.Key) != 1) return false;
            if (currNode.RightSon != null && currNode.Key.CompareTo(rightSon.Key) != -1) return false;

            return CheckRuleRec(leftSon) && CheckRuleRec(rightSon);
        }
    }
}