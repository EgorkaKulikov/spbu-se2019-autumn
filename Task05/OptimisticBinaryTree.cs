using System.Threading;

namespace Task05
{
    public class OptimisticBinaryTree
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

            while(tempNode != null) 
            {
                if (tempNode.key == key)
                { 
                    return tempNode.value;
                }
                else if (tempNode.key > key)
                {
                    if (tempNode.left != null)
                    {
                        tempNode = tempNode.left;
                    }
                    else
                    {
                        break;
                    }
                }
                else if (tempNode.key < key)
                {
                    if (tempNode.right != null)
                    {
                        tempNode = tempNode.right;
                    }
                    else
                    {
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

            var tempNode = root;
            while (true)
            {
                if (tempNode.key > key)
                {
                    if (tempNode.left == null)
                    {
                        tempNode.mutex.WaitOne();
                        if (tempNode.left == null)
                        {
                            var newNode = new Node(key, value);
                            tempNode.left = newNode;
                            tempNode.mutex.ReleaseMutex();
                            return;
                        }
                        tempNode.mutex.ReleaseMutex();
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
                        tempNode.mutex.WaitOne();
                        if (tempNode.right == null)
                        {
                            var newNode = new Node(key, value);
                            tempNode.right = newNode;
                            tempNode.mutex.ReleaseMutex();
                            return;
                        }
                        tempNode.mutex.ReleaseMutex();
                    }
                    else
                    {
                        tempNode = tempNode.right;
                    }
                }
                if (tempNode.key == key)
                {
                    return;
                }
            }
        }
    }
} 