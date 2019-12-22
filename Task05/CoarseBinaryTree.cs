using System.Threading;

namespace Task05
{
    public class CoarseBinaryTree
    {
        public class Node
        {
            public int key;
            public int value;
            public Node left = null;
            public Node right = null;

            public Node(int key, int value)
            {
                this.key = key;
                this.value = value;
            }
        }
        public Node root = null;
        public Mutex mutex = new Mutex();

        public bool isBinaryTree(Node node)
        {
            if (node == null)
                return true;
            if (node.left != null)
                if (node.left.value > node.value)
                    return false;

            if (node.right != null)
                if (node.right.value < node.value)
                    return false;

            return isBinaryTree(node.left) && isBinaryTree(node.right);
        }



        public int? find(int key)
        {
            mutex.WaitOne();
            var tempNode = root;
            while(tempNode != null) {
                if (tempNode.key == key)
                {
                    mutex.ReleaseMutex();
                    return tempNode.value;
                }
                else if (tempNode.key > key)
                {
                    tempNode = tempNode.left;
                }
                else if (tempNode.key < key)
                {
                    tempNode = tempNode.right;
                }
            }

            mutex.ReleaseMutex();
            return null;
        }

        public void insert(int key, int value)
        {
            mutex.WaitOne();
            if (root == null)
            {
                root = new Node(key, value);
                mutex.ReleaseMutex();
                return;
            }

            var tempNode = root;
            while (true)
            {
                if (tempNode.key > key)
                {
                    if (tempNode.left == null)
                    {
                        var newNode = new Node(key, value);
                        tempNode.left = newNode;
                        mutex.ReleaseMutex();
                        return;
                    }
                    else
                    {
                        tempNode = tempNode.left;
                    }
                }
                if (tempNode.key < key)
                {
                    if (tempNode.right == null)
                    {
                        var newNode = new Node(key, value);
                        tempNode.right = newNode;
                        mutex.ReleaseMutex();
                        return;
                    }
                    else
                    {
                        tempNode = tempNode.right;
                    }
                }
                if (tempNode.key == key)
                {
                    tempNode.value = value;
                    mutex.ReleaseMutex();
                    return;
                }
            }
        }
    }
} 