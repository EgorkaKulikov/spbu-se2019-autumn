using System.Collections.Generic;
using System.Threading;

namespace Task05
{
    public class BSTFineGrained<T> :BST<T>
    {
        public class FGNode<T> : AbstractNode<T>
        {
            public Mutex mutex;
            public int key;
            public T value;
            public FGNode<T> left;
            public FGNode<T> right;
            public FGNode<T> parent;

            public FGNode(FGNode<T> parent, int key, T value)
            {
                this.key = key;
                this.value = value;
                left = null;
                right = null;
                this.parent = parent;
                mutex = new Mutex(false);
            }

            ~FGNode() {
                mutex.Close();
            }
        }

        public BSTFineGrained()
        {
        }
        public BSTFineGrained(IEnumerable<(int, T)> elements)
        {
            foreach (var pair in elements)
            {
                Insert(pair.Item1, pair.Item2);
            }
        }
        
        protected internal FGNode<T> root = null;

        Mutex rootInsertCheck = new Mutex(false);

        public new void Insert(int key, T value)
        {
            rootInsertCheck.WaitOne();
            if (root == null)
            {
                root = new FGNode<T>(null, key, value);
                rootInsertCheck.ReleaseMutex();
                return;
            }

            rootInsertCheck.ReleaseMutex();
            UpdateTree(null, root, key, value, true);
        }

        protected void UpdateTree(FGNode<T> parent, FGNode<T> currentNode, int key, T value, bool leftChild)
        {
            if (currentNode == null)
            {
                if (leftChild)
                {
                    parent.left = new FGNode<T>(parent, key, value);
                }
                else
                {
                    parent.right = new FGNode<T>(parent, key, value);
                }

                parent.mutex.ReleaseMutex();
            }
            else
            {
                if (currentNode.key == key)
                {
                    parent?.mutex.ReleaseMutex();
                    return;
                }

                currentNode.mutex.WaitOne();
                parent?.mutex.ReleaseMutex();
                if (currentNode.key > key)
                {
                    UpdateTree(currentNode, currentNode.left, key, value, true);
                }
                else
                {
                    UpdateTree(currentNode, currentNode.right, key, value, false);
                }
            }
        }

        public new T Search(int key)
        {
            root?.mutex.WaitOne();
            var node = FindInTree(root, key);
            node?.mutex.ReleaseMutex();
            
            if (node == null)
            {
                return default(T);
            }
            
            return node.value;
        }

        private FGNode<T> FindInTree(FGNode<T> currentNode, int key)
        {
            if (currentNode == null)
            {
                return null;
            }

            if (currentNode.key == key)
            {
                return currentNode;
            }
            
            if (currentNode.key > key)
            {
                currentNode.left?.mutex.WaitOne();
                currentNode.mutex.ReleaseMutex();
                return FindInTree(currentNode.left, key);
            }
            currentNode.right?.mutex.WaitOne();
            currentNode.mutex.ReleaseMutex();
            return FindInTree(currentNode.right, key);
        }

        public new void Delete(int key)
        {
            root?.mutex.WaitOne(); //will be released if we are not searching for root, and node will be locked
            FGNode<T> node = FindInTree(root, key);
            if (node == null)
                return;
            
            node.right?.mutex.WaitOne();
            FGNode<T> replacement = FindLeftMost(node.right);

            bool foundLeftMost = true;
            if (replacement == null)
            {
                node.left?.mutex.WaitOne();
                replacement = FindRightMost(node.left);
                foundLeftMost = false;
            }

            if (replacement == null)
            {
                if (node.parent == null) //is root
                {
                    root.mutex.ReleaseMutex();
                    root = null;
                }
                else
                {
                    if (node.parent.left != null)
                    {
                        if (node.parent.left.key == node.key)
                        {
                            //left son
                            node.parent.left = null;
                        }
                        else
                        {
                            node.parent.right = null;
                        }
                    }
                    else node.parent.right = null;
                    node.mutex.ReleaseMutex();
                }
            }
            else
            {
                //replacement parent is not null, because it is not root
                //remove replacement with fixing links
                if (replacement.parent.left != null)
                {
                    if (replacement.parent.left.key == replacement.key)
                    {
                        //left son
                        if (foundLeftMost)
                        {
                            replacement.parent.left = replacement.right;
                        }
                        else
                        {
                            replacement.parent.left = replacement.left;
                        }
                    }
                    else
                    {
                        if (foundLeftMost)
                        {
                            replacement.parent.right = replacement.right;
                        }
                        else
                        {
                            replacement.parent.right = replacement.left;
                        }
                    }
                }
                else
                {
                    if (foundLeftMost)
                    {
                        replacement.parent.right = replacement.right;
                    }
                    else
                    {
                        replacement.parent.right = replacement.left;
                    }
                }

                //insert as root
                if (node.parent == null)
                {
                    root.key = replacement.key;
                    root.value = replacement.value;
                }
                else
                {
                    node.key = replacement.key;
                    node.value = replacement.value;
                }
                replacement.mutex.ReleaseMutex();
                node.mutex.ReleaseMutex();
            }
        }

        private FGNode<T> FindRightMost(FGNode<T> node)
        {
            if (node == null)
            {
                return null;
            }

            if (node.right == null)
            {
                return node;
            }
            
            node.right.mutex.WaitOne();
            node.mutex.ReleaseMutex();
            return FindRightMost(node.right);
        }

        private FGNode<T> FindLeftMost(FGNode<T> node)
        {
            if (node == null)
            {
                return null;
            }

            if (node.left == null)
            {
                return node;
            }

            node.left?.mutex.WaitOne();
            node.mutex.ReleaseMutex();
            return FindLeftMost(node.left);
        }
    }
}