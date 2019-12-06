using System;
using System.Threading;
using System.Threading.Tasks;

namespace Task05
{
    public class Program
    {
        static void Main(string[] args)
        {
            Console.Read();
        }
    }
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
                root = InerInsert(root, newItem);
            Console.WriteLine("Inserted {0}", value);
            TreeMutex.ReleaseMutex();
        }
        private Node InerInsert(Node current, Node n)
        {
            if (current == null)
            {
                current = n;
                return current;
            }
            else if (n.value < current.value)
            {
                current.left = InerInsert(current.left, n);
                current = balance_tree(current);
            }
            else if (n.value > current.value)
            {
                current.right = InerInsert(current.right, n);
                current = balance_tree(current);
            }
            return current;
        }
        private Node balance_tree(Node current)
        {
            int b_factor = balance_factor(current);
            if (b_factor > 1)
            {
                if (balance_factor(current.left) > 0)
                    current = RotateLL(current);
                else
                    current = RotateLR(current);
            }
            else if (b_factor < -1)
            {
                if (balance_factor(current.right) > 0)
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
                    if (balance_factor(current) == -2)
                    {
                        if (balance_factor(current.right) <= 0)
                            current = RotateRR(current);
                        else
                            current = RotateRL(current);
                    }
                }
                else if (target > current.value)
                {
                    current.right = InerDelete(current.right, target);
                    if (balance_factor(current) == 2)
                    {
                        if (balance_factor(current.left) >= 0)
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
                        if (balance_factor(current) == 2)
                        {
                            if (balance_factor(current.left) >= 0)
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
        private int max(int l, int r)
        {
            return l > r ? l : r;
        }
        private int getHeight(Node current)
        {
            int height = 0;
            if (current != null)
            {
                int l = getHeight(current.left);
                int r = getHeight(current.right);
                height = max(l, r) + 1;
            }
            return height;
        }
        private int balance_factor(Node current)
        {
            int l = getHeight(current.left);
            int r = getHeight(current.right);
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
    public class LazyAVL
    {
        private Mutex TreeMutex = new Mutex();
        private class Node
        {
            public int value;
            public Node left;
            public Node right;
            public bool marked;
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
                root = InerInsert(root, newItem);
            Console.WriteLine("Inserted {0}", value);
            TreeMutex.ReleaseMutex();
        }
        private Node InerInsert(Node current, Node n)
        {
            if (current == null)
            {
                current = n;
                return current;
            }
            else if (n.value < current.value)
            {
                current.left = InerInsert(current.left, n);
                current = balance_tree(current);
            }
            else if (n.value > current.value)
            {
                current.right = InerInsert(current.right, n);
                current = balance_tree(current);
            }
            return current;
        }
        private Node balance_tree(Node current)
        {
            int b_factor = balance_factor(current);
            if (b_factor > 1)
            {
                if (balance_factor(current.left) > 0)
                    current = RotateLL(current);
                else
                    current = RotateLR(current);
            }
            else if (b_factor < -1)
            {
                if (balance_factor(current.right) > 0)
                    current = RotateRL(current);
                else
                    current = RotateRR(current);
            }
            return current;
        }
        public bool Delete(int target)
        {
            while (true)
            {
                Node pred = root;
                Node curr;
                if (target < pred.value)
                    curr = pred.left;
                else
                    curr = pred.right;
                while (target != curr.value)
                {
                    pred = curr;
                    if (target < pred.value)
                        curr = pred.left;
                    else
                        curr = pred.right;
                }
                TreeMutex.WaitOne();
                try
                {
                    if (!pred.marked && !curr.marked && (pred.left == curr || pred.right == curr))
                    {
                        if (curr.value == target)
                        {
                            curr.marked = true;
                            return true;
                        }
                        else
                            return false;
                    }
                }
                finally
                {
                    TreeMutex.ReleaseMutex();
                }
            }
        }
        public void RealDelete()
        {
            InerRealDelete(root);
        }
        private void InerRealDelete(Node current)
        {
            if (current != null)
            {
                InerRealDelete(current.left);
                if (current.marked)
                    InerDelete(current, current.value);
                InerRealDelete(current.right);
            }
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
                    if (balance_factor(current) == -2)
                    {
                        if (balance_factor(current.right) <= 0)
                            current = RotateRR(current);
                        else
                            current = RotateRL(current);
                    }
                }
                else if (target > current.value)
                {
                    current.right = InerDelete(current.right, target);
                    if (balance_factor(current) == 2)
                    {
                        if (balance_factor(current.left) >= 0)
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
                        if (balance_factor(current) == 2)
                        {
                            if (balance_factor(current.left) >= 0)
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
            Node ans = InerFind(key, root);
            if (ans != null)
                return true;
            else
                return false;
        }
        private Node InerFind(int target, Node current)
        {
            if (current != null)
            {
                if (current.value > target)
                    return InerFind(target, current.left);
                if (current.value < target)
                    return InerFind(target, current.right);
                if(!current.marked)
                    return current;
                return null;
            }
            else
                return null;
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
                if (!current.marked)
                {
                    ans += current.value.ToString();
                    ans += " ";
                }
                ans = InerDisplayTree(current.right, ans);
            }
            return ans;
        }
        private int max(int l, int r)
        {
            return l > r ? l : r;
        }
        private int getHeight(Node current)
        {
            int height = 0;
            if (current != null)
            {
                int l = getHeight(current.left);
                int r = getHeight(current.right);
                height = max(l, r) + 1;
            }
            return height;
        }
        private int balance_factor(Node current)
        {
            int l = getHeight(current.left);
            int r = getHeight(current.right);
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