using System;
using System.Threading;

namespace Task05
{
    public class CoarseAVL
    {
        private static readonly Mutex TreeMutex = new Mutex();
        private class Node
        {
            public int Value;
            public Node Left;
            public Node Right;
            public Node(int value)
            {
                this.Value = value;
            }
        }

        private Node _root;

        public void Add(int value)
        {
            TreeMutex.WaitOne();
            var newItem = new Node(value);
            _root = (_root == null) ? newItem : InnerInsert(_root, newItem);
            TreeMutex.ReleaseMutex();
        }
        private static Node InnerInsert(Node current, Node n)
        {
            if (current == null)
            {
                current = n;
                return current;
            }
            if (n.Value < current.Value)
            {
                current.Left = InnerInsert(current.Left, n);
                current = BalanceTree(current);
            }
            else if (n.Value > current.Value)
            {
                current.Right = InnerInsert(current.Right, n);
                current = BalanceTree(current);
            }
            return current;
        }
        private static Node BalanceTree(Node current)
        {
            var bFactor = BalanceFactor(current);
            if (bFactor > 1)
            {
                current = BalanceFactor(current.Left) > 0 ? RotateLL(current) : RotateLR(current);
            }
            else if (bFactor < -1)
            {
                current = BalanceFactor(current.Right) > 0 ? RotateRL(current) : RotateRR(current);
            }
            return current;
        }
        public void Delete(int target)
        {
            TreeMutex.WaitOne();
            _root = InnerDelete(_root, target);
            TreeMutex.ReleaseMutex();
        }
        private static Node InnerDelete(Node current, int target)
        {
            if (current == null)
                return null;
            if (target < current.Value)
            {
                    current.Left = InnerDelete(current.Left, target);
                    if (BalanceFactor(current) == -2)
                    {
                        current = BalanceFactor(current.Right) <= 0 ? RotateRR(current) : RotateRL(current);
                    }
            }
            else if (target > current.Value)
            {
                    current.Right = InnerDelete(current.Right, target);
                    if (BalanceFactor(current) != 2) return current;
                    current = BalanceFactor(current.Left) >= 0 ? RotateLL(current) : RotateLR(current);
            }
            else
            {
                    if (current.Right != null)
                    {
                        var parent = current.Right;
                        while (parent.Left != null)
                        {
                            parent = parent.Left;
                        }
                        current.Value = parent.Value;
                        current.Right = InnerDelete(current.Right, parent.Value);
                        if (BalanceFactor(current) == 2)
                        {
                            current = BalanceFactor(current.Left) >= 0 ? RotateLL(current) : RotateLR(current);
                        }
                    }
                    else
                        return current.Left;
            }
            return current;
        }
        public bool Find(int key)
        {
            TreeMutex.WaitOne();
            try
            {
                return InnerFind(key, _root);
            }
            finally
            {
                TreeMutex.ReleaseMutex();
            }
        }
        private static bool InnerFind(int target, Node current)
        {
            if (current != null)
            {
                if (current.Value > target)
                    return InnerFind(target, current.Left);
                if (current.Value < target)
                    return InnerFind(target, current.Right);
                return true;
            }
            else
                return false;
        }
        public string DisplayTree()
        {
            if (_root == null)
                return "Tree is empty";
            string ans = "";
            ans = InnerDisplayTree(_root, ans);
            return ans;
        }
        private static string InnerDisplayTree(Node current, string ans)
        {
            if (current != null)
            {
                ans = InnerDisplayTree(current.Left, ans);
                ans += current.Value.ToString();
                ans += " ";
                ans = InnerDisplayTree(current.Right, ans);
            }
            return ans;
        }
        private static int Max(int l, int r)
        {
            return l > r ? l : r;
        }
        private static int GetHeight(Node current)
        {
            int height = 0;
            if (current != null)
            {
                int l = GetHeight(current.Left);
                int r = GetHeight(current.Right);
                height = Max(l, r) + 1;
            }
            return height;
        }
        private static int BalanceFactor(Node current)
        {
            int l = GetHeight(current.Left);
            int r = GetHeight(current.Right);
            return l - r;
        }
        private static Node RotateRR(Node parent)
        {
            Node pivot = parent.Right;
            parent.Right = pivot.Left;
            pivot.Left = parent;
            return pivot;
        }
        private static Node RotateLL(Node parent)
        {
            Node pivot = parent.Left;
            parent.Left = pivot.Right;
            pivot.Right = parent;
            return pivot;
        }
        private static Node RotateLR(Node parent)
        {
            Node pivot = parent.Left;
            parent.Left = RotateRR(pivot);
            return RotateLL(parent);
        }
        private static Node RotateRL(Node parent)
        {
            Node pivot = parent.Right;
            parent.Right = RotateLL(pivot);
            return RotateRR(parent);
        }
        ~CoarseAVL()
        {
            TreeMutex.Dispose();
        }
    }
}