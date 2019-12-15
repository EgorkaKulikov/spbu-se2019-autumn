using System;
using System.Threading;

namespace Task05
{
    public class CoarseAVL
    {
        private Mutex TreeMutex = new Mutex();
        private class Node
        {
            public int value;
            public Node left;
            public Node right;
            public Node(int value)
            {
                this.value = value;
            }
        }
        Node root;

        public void Add(int value)
        {
            TreeMutex.WaitOne();
            Node newItem = new Node(value);
            if (root == null)
                root = newItem;
            else
                root = InnerInsert(root, newItem);
            Console.WriteLine("Inserted {0}", value);
            TreeMutex.ReleaseMutex();
        }
        private Node InnerInsert(Node current, Node n)
        {
            if (current == null)
            {
                current = n;
                return current;
            }
            else if (n.value < current.value)
            {
                current.left = InnerInsert(current.left, n);
                current = BalanceTree(current);
            }
            else if (n.value > current.value)
            {
                current.right = InnerInsert(current.right, n);
                current = BalanceTree(current);
            }
            return current;
        }
        private Node BalanceTree(Node current)
        {
            int b_factor = BalanceFactor(current);
            if (b_factor > 1)
            {
                if (BalanceFactor(current.left) > 0)
                    current = RotateLL(current);
                else
                    current = RotateLR(current);
            }
            else if (b_factor < -1)
            {
                if (BalanceFactor(current.right) > 0)
                    current = RotateRL(current);
                else
                    current = RotateRR(current);
            }
            return current;
        }
        public void Delete(int target)
        {
            TreeMutex.WaitOne();
            root = InerDelete(root, target);
            TreeMutex.ReleaseMutex();
        }
        private Node InerDelete(Node current, int target)
        {
            Node parent;
            if (current == null)
                return null;
            else
            {
                if (target < current.value)
                {
                    current.left = InerDelete(current.left, target);
                    if (BalanceFactor(current) == -2)
                    {
                        if (BalanceFactor(current.right) <= 0)
                            current = RotateRR(current);
                        else
                            current = RotateRL(current);
                    }
                }
                else if (target > current.value)
                {
                    current.right = InerDelete(current.right, target);
                    if (BalanceFactor(current) == 2)
                    {
                        if (BalanceFactor(current.left) >= 0)
                            current = RotateLL(current);
                        else
                            current = RotateLR(current);
                    }
                }
                else
                {
                    if (current.right != null)
                    {
                        parent = current.right;
                        while (parent.left != null)
                        {
                            parent = parent.left;
                        }
                        current.value = parent.value;
                        current.right = InerDelete(current.right, parent.value);
                        if (BalanceFactor(current) == 2)
                        {
                            if (BalanceFactor(current.left) >= 0)
                                current = RotateLL(current);
                            else
                                current = RotateLR(current);
                        }
                    }
                    else
                        return current.left;
                }
            }
            return current;
        }
        public bool Find(int key)
        {
            TreeMutex.WaitOne();
            try
            {
                return InerFind(key, root);
            }
            finally
            {
                TreeMutex.ReleaseMutex();
            }
        }
        private bool InerFind(int target, Node current)
        {
            if (current != null)
            {
                if (current.value > target)
                    return InerFind(target, current.left);
                if (current.value < target)
                    return InerFind(target, current.right);
                return true;
            }
            else
                return false;
        }
        public string DisplayTree()
        {
            if (root == null)
                return "Tree is empty";
            string ans = "";
            ans = InerDisplayTree(root, ans);
            return ans;
        }
        private string InerDisplayTree(Node current, string ans)
        {
            if (current != null)
            {
                ans = InerDisplayTree(current.left, ans);
                ans += current.value.ToString();
                ans += " ";
                ans = InerDisplayTree(current.right, ans);
            }
            return ans;
        }
        private int Max(int l, int r)
        {
            return l > r ? l : r;
        }
        private int GetHeight(Node current)
        {
            int height = 0;
            if (current != null)
            {
                int l = GetHeight(current.left);
                int r = GetHeight(current.right);
                height = Max(l, r) + 1;
            }
            return height;
        }
        private int BalanceFactor(Node current)
        {
            int l = GetHeight(current.left);
            int r = GetHeight(current.right);
            return l - r;
        }
        private Node RotateRR(Node parent)
        {
            Node pivot = parent.right;
            parent.right = pivot.left;
            pivot.left = parent;
            return pivot;
        }
        private Node RotateLL(Node parent)
        {
            Node pivot = parent.left;
            parent.left = pivot.right;
            pivot.right = parent;
            return pivot;
        }
        private Node RotateLR(Node parent)
        {
            Node pivot = parent.left;
            parent.left = RotateRR(pivot);
            return RotateLL(parent);
        }
        private Node RotateRL(Node parent)
        {
            Node pivot = parent.right;
            parent.right = RotateLL(pivot);
            return RotateRR(parent);
        }
    }
}
