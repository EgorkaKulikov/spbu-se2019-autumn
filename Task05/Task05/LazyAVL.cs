using System;
using System.Threading;

namespace Task05
{
    public class LazyAVL
    {
        private static readonly Mutex TreeMutex = new Mutex();
        private class Node
        {
            public int Value;
            public Node Left;
            public Node Right;
            public bool Marked;
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
            _root = _root == null ? newItem : InnerInsert(_root, newItem);
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
        public bool Delete(int target)
        {
            if (_root == null)
                return false;
            while (true)
            {
                var previous = _root;
                if (previous.Value == target)
                {
                    TreeMutex.WaitOne();
                    try
                    {
                        if (previous.Value == target)
                        {
                            previous.Marked = true;
                            return true;
                        }
                    }
                    finally
                    {
                        TreeMutex.ReleaseMutex();
                    }
                }
                var current = target < previous.Value ? previous.Left : previous.Right;
                while (current != null && target != current.Value)
                {
                    previous = current;
                    current = target < previous.Value ? previous.Left : previous.Right;
                }

                if (current == null)
                {
                    return false;
                }
                TreeMutex.WaitOne();
                try
                {
                    if (!current.Marked &&
                        (previous.Left == current || previous.Right == current))
                    {
                        if (current.Value == target)
                        {
                            current.Marked = true;
                            return true;
                        }
                    }
                    else
                        return false;
                }
                finally
                {
                    TreeMutex.ReleaseMutex();
                }
            }
        }
        public void RealDelete()
        {
            TreeMutex.WaitOne();
            var a = true;
            if (_root != null)
            {
                while (a) a = InnerRealDelete(_root);
            }
            else
                TreeMutex.ReleaseMutex();
        }
        private bool InnerRealDelete(Node current)
        {
            if (current == null) return false;
            if (current.Marked)
            {
                _root = InnerDelete(_root, current.Value);
                return true;
            }
            return InnerRealDelete(current.Left) || InnerRealDelete(current.Right);
        }
        private static Node InnerDelete(Node current, int target)
        {
            if (target < current.Value)
            {
                current.Left = InnerDelete(current.Left, target);
                if (BalanceFactor(current) != -2) return current;
                current = BalanceFactor(current.Right) <= 0 ? RotateRR(current) : RotateRL(current);
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
                        current = BalanceFactor(current.Left) >= 0 ? RotateLL(current) : RotateLR(current);
                }
                else
                    return current.Left;
            }
            return current;
        }
        public bool Find(int key)
        {
            Node ans = InnerFind(key, _root);
            if (ans != null)
                return true;
            else
                return false;
        }
        private static Node InnerFind(int target, Node current)
        {
            if (current != null)
            {
                if (current.Value > target)
                    return InnerFind(target, current.Left);
                if (current.Value < target)
                    return InnerFind(target, current.Right);
                if (!current.Marked)
                    return current;
                return null;
            }
            else
                return null;
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
                if (!current.Marked)
                {
                    ans += current.Value.ToString();
                    ans += " ";
                }
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
        ~LazyAVL()
        {
            TreeMutex.Dispose();
        }
    }
}