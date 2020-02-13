using System.Threading;

namespace Task05
{
    public class FineBinaryTree
    {
        public class Node
        {
            public int key;
            public int value;
            public Node left = null;
            public Node right = null;
            public Mutex mutex = new Mutex();

            public Node(int key, int value)
            {
                this.key = key;
                this.value = value;
            }
        }
        
        private Mutex rootMutex = new Mutex();
        public Node root = null;

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

        public bool mutexesRealesed(Node node)
        {
            if (node != null)
            {
                var isUnlocked = node.mutex.WaitOne(0);
                if (!isUnlocked)
                    return false;

                node.mutex.ReleaseMutex();

                return mutexesRealesed(node.left) && mutexesRealesed(node.right);
            }

            return true;
        }
        public int? find(int key)
        {
            if (root == null)
                return null;

            var tempNode = root;
            tempNode.mutex.WaitOne();

            while(tempNode != null) 
            {
                if (tempNode.key == key)
                {
                    int val = tempNode.value;
                    tempNode.mutex.ReleaseMutex();
                    return val;
                }
                else if (tempNode.key > key)
                {
                    if (tempNode.left != null)
                    {
                        var tmp = tempNode;
                        tempNode.left.mutex.WaitOne();
                        tempNode = tempNode.left;
                        tmp.mutex.ReleaseMutex();
                    }
                    else
                    {
                        tempNode.mutex.ReleaseMutex();
                        break;
                    }
                }
                else if (tempNode.key < key)
                {
                    if (tempNode.right != null)
                    {
                        var tmp = tempNode;
                        tempNode.right.mutex.WaitOne();
                        tempNode = tempNode.right;
                        tmp.mutex.ReleaseMutex();
                    }
                    else
                    {
                        tempNode.mutex.ReleaseMutex();
                        break;
                    }
                }
            }

            return null;
        }

        public void insert(int key, int value)
        {
            if (root == null)
            {
                rootMutex.WaitOne();
                if (root == null)
                {
                    root = new Node(key, value);
                    rootMutex.ReleaseMutex();
                    return;
                }
                rootMutex.ReleaseMutex();
            }

            root.mutex.WaitOne();
            var tempNode = root;
            while (true)
            {
                if (tempNode.key > key)
                {
                    if (tempNode.left == null)
                    {
                        var newNode = new Node(key, value);
                        tempNode.left = newNode;
                        tempNode.mutex.ReleaseMutex();
                        return;
                    }
                    else
                    {
                        var tmp = tempNode;
                        tempNode.left.mutex.WaitOne();
                        tempNode = tempNode.left;
                        tmp.mutex.ReleaseMutex();
                    }
                }
                if (tempNode.key < key)
                {
                    if (tempNode.right == null)
                    {
                        var newNode = new Node(key, value);
                        tempNode.right = newNode;
                        tempNode.mutex.ReleaseMutex();
                        return;
                    }
                    else
                    {
                        var tmp = tempNode;
                        tempNode.right.mutex.WaitOne();
                        tempNode = tempNode.right;
                        tmp.mutex.ReleaseMutex();
                    }
                }
                if (tempNode.key == key)
                {
                    tempNode.value = value;
                    tempNode.mutex.ReleaseMutex();
                    return;
                }
            }
        }
    }
} 