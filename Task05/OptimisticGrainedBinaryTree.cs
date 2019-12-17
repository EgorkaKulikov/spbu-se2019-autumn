using System.Threading;

namespace Task05
{
    public class OptimisticGrainedBinaryTree
    {
        private readonly Mutex _mutex = new Mutex();
        public BinaryTreeNode Root;

        public bool Find(int key)
        {
            return FindNode(Root, key);
        }
        
        public void Insert(int key)
        {
            InsertNode(new BinaryTreeNode(key));
        }
        
        private bool FindNode(BinaryTreeNode? node, int key)
        {
            while (node != null)
            {
                if (node.Key == key) return true;

                node = node.Key < key ? node.Right : node.Left;
            }

            return false;
        }

        private void InsertNode(BinaryTreeNode node)
        {
            if (Root == null)
            {
                _mutex.WaitOne();
                if (Root == null)
                {
                    Root = node;
                    _mutex.ReleaseMutex();
                    return;
                }

                _mutex.ReleaseMutex();
                InsertNode(node);
            }

            var currentNode = Root;

            while (currentNode != null)
            {
                if (currentNode.Key == node.Key) break;

                if (node.Key > currentNode.Key)
                {
                    if (currentNode.Right == null)
                    {
                        currentNode.NodeMutex.WaitOne();
                        if (currentNode.Right == null)
                        {
                            node.Parent = currentNode;
                            currentNode.Right = node;
                            currentNode.NodeMutex.ReleaseMutex();
                            break;
                        }

                        currentNode.NodeMutex.ReleaseMutex();
                        InsertNode(node);
                    }
                    else
                    {
                        currentNode = currentNode.Right;
                    }
                }
                else if (node.Key < currentNode.Key)
                {
                    if (currentNode.Left == null)
                    {
                        currentNode.NodeMutex.WaitOne();
                        if (currentNode.Left == null)
                        {
                            node.Parent = currentNode;
                            currentNode.Left = node;
                            currentNode.NodeMutex.ReleaseMutex();
                            break;
                        }

                        currentNode.NodeMutex.ReleaseMutex();
                        InsertNode(node);
                    }
                    else
                    {
                        currentNode = currentNode.Left;
                    }
                }
            }
        }
    }
}