using System;
using System.Collections.Generic;

namespace Task05
{
    /*abstract*/
    public class BST<T>
    {
        protected internal Node<T> root = null;

        public void Show()
        {
            ShowNode(root, 0);
        }

        public BST() 
        {
        }

        public BST(IEnumerable<(int, T)> elements)
        {
            foreach (var pair in elements)
            {
                Insert(pair.Item1, pair.Item2);
            }
        }
        private static void ShowNode(Node<T> node, int space)
        {
            if (node == null)
                return;

            // Increase distance between levels  
            space += 10;

            // Process right child first  
            ShowNode(node.right, space);

            // Print current node after space  
            // count  
            Console.Write("\n");
            for (int i = 10; i < space; i++)
                Console.Write(" ");
            Console.Write(node.key + "\n");

            // Process left child  
            ShowNode(node.left, space);
        }

        public virtual void Insert(int key, T value)
        {
            UpdateTree(null, root, key, value, true);
        }

        protected void UpdateTree(Node<T> parent, Node<T> currentNode, int key, T value, bool leftChild)
        {
            if (currentNode == null)
            {
                if (parent == null)
                {
                    root = new Node<T>(parent, key, value);
                    return;
                }

                if (leftChild)
                {
                    parent.left = new Node<T>(parent, key, value);
                }
                else
                {
                    parent.right = new Node<T>(parent, key, value);
                }
            }
            else
            {
                if (currentNode.key == key) return;
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

        public virtual T Search(int key)
        {
            var node = FindInTree(root, key);
            if (node == null)
            {
                return default(T);
            }
            else
            {
                return node.value;
            }
        }

        public virtual void Delete(int key)
        {
            Node<T> node = FindInTree(root, key);
            if (node == null)
                return;


            var replacement = FindLeftMost(node.right);
            bool foundLeftMost = true;

            if (replacement == null)
            {
                replacement = FindRightMost(node.left);
                foundLeftMost = false;
            }

            if (replacement == null)
            {
                if (node.parent == null) //is root
                {
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
            }
        }

        private Node<T> FindRightMost(Node<T> node)
        {
            if (node == null)
            {
                return null;
            }

            if (node.right == null)
            {
                return node;
            }

            return FindRightMost(node.right);
        }

        private Node<T> FindLeftMost(Node<T> node)
        {
            if (node == null)
            {
                return null;
            }

            if (node.left == null)
            {
                return node;
            }

            return FindLeftMost(node.left);
        }

        private Node<T> FindInTree(Node<T> currentNode, int key)
        {
            if (currentNode == null)
            {
                return null;
            }

            if (currentNode.key == key)
            {
                return currentNode;
            }
            else
            {
                if (currentNode.key > key)
                {
                    return FindInTree(currentNode.left, key);
                }
                else
                {
                    return FindInTree(currentNode.right, key);
                }
            }
        }
    }
}