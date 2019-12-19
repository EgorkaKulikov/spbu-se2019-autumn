using System;
using System.Collections.Generic;

namespace Task05
{
    public class Utils
    {
        private static void AddValuesToSet<T>(HashSet<T> hashSet, AbstractNode<T> node)
        {
            if (node == null) return;
            hashSet.Add(node.value);
            AddValuesToSet(hashSet, node.left);
            AddValuesToSet(hashSet, node.right);
        }

        public static bool ContentEquals<T>(BST<T> bst1, BST<T> bst2)
        {
            HashSet<T> values1 = new HashSet<T>();
            HashSet<T> values2 = new HashSet<T>();
            AddValuesToSet(values1, bst1.root);
            AddValuesToSet(values2, bst2.root);
            bool result = true;
            foreach (var value in values1)
            {
                result = result && values2.Contains(value);
            }
            foreach (var value in values2)
            {
                result = result && values1.Contains(value);
            }

            return result;
        }

        private static bool ValidateNode<T>(AbstractNode<T> node)
        {
            if (node == null) return true;
            var middleKey = node.key;
            var leftKey = node.key - 1;
            if (node.left != null)
            {
                leftKey = node.left.key;
            }

            var rightKey = node.key + 1;
            if (node.right != null)
            {
                rightKey = node.right.key;
            }

            return (leftKey < middleKey) && (middleKey < rightKey)
                                         && ValidateNode(node.left) && ValidateNode(node.right);
        }

        public static bool ValidateStructure<T>(BST<T> bst)
        {
            return ValidateNode(bst.root);
        }
    }
}