using System;
using System.Threading;

namespace Task05
{
    public class FineGrainedTree<TKey, TValue> : IBinarySearchTree<TKey, TValue>
        where TKey : IComparable
        where TValue : struct
    {
        private readonly Mutex _rootMtx = new Mutex();
        private Node<TKey, TValue> _root;

        public bool IsEmpty()
        {
            return _root == null;
        }

        public void Insert(TKey key, TValue value)
        {
            _rootMtx.WaitOne();
            var curNode = _root;
            
            if (_root == null)
            {
                _root = new Node<TKey, TValue>(key, value, null);
                _rootMtx.ReleaseMutex();
                return;
            }
            _rootMtx.ReleaseMutex();
            Node<TKey, TValue> parent = null;
            curNode?.NodeMtx.WaitOne();
            while (curNode != null && !curNode.Key.Equals(key))
            {
                parent?.NodeMtx.ReleaseMutex();
                parent = curNode;
                var cmpResult = curNode.Key.CompareTo(key);
                if (cmpResult > 0)
                {
                    curNode = curNode.Left ?? new Node<TKey, TValue>(key, value, curNode);
                    parent.Left = curNode;
                }
                else
                {
                    curNode = curNode.Right ?? new Node<TKey, TValue>(key, value, curNode);
                    parent.Right = curNode;
                }
                curNode.NodeMtx.WaitOne();
            }

            if (curNode != null)
            {
                curNode.Value = value;
            }
            parent?.NodeMtx.ReleaseMutex();
            curNode?.NodeMtx.ReleaseMutex();
        }

        public TValue? Find(TKey key)
        {
            var curNode = _root;
            Node<TKey, TValue> parent = null;
            curNode?.NodeMtx.WaitOne();
            while (curNode != null && !curNode.Key.Equals(key))
            {
                parent?.NodeMtx.ReleaseMutex();
                parent = curNode;
                var curKey = curNode.Key;
                curNode = curKey.CompareTo(key) > 0 ? curNode.Left : curNode.Right;
                curNode?.NodeMtx.WaitOne();
            }

            var value = curNode?.Value;
            parent?.NodeMtx.ReleaseMutex();
            curNode?.NodeMtx.ReleaseMutex();
            
            return value;
        }

        public void Remove(TKey key)
        {

            var remNode = _root;
            Node<TKey, TValue> parentNode = null;
            remNode?.NodeMtx.WaitOne();
            while (remNode != null && !remNode.Key.Equals(key))
            {
                parentNode?.NodeMtx.ReleaseMutex();
                parentNode = remNode;
                var curKey = remNode.Key;
                remNode = curKey.CompareTo(key) > 0 ? remNode.Left : remNode.Right;
                remNode?.NodeMtx.WaitOne();
            }

            if (remNode == null)
            {
                parentNode?.NodeMtx.ReleaseMutex();
                return;
            }
            
            if (remNode.Left != null)
            {
                if (remNode.Right != null)
                {
                    CopyNextNode(remNode);
                }
                else
                {
                    RemoveNode(remNode, remNode.Left);
                }
            }
            else
            {
                RemoveNode(remNode, remNode.Right);
            }
            parentNode?.NodeMtx.ReleaseMutex();
            remNode.NodeMtx.ReleaseMutex();
        }

        private void CopyNextNode(Node<TKey, TValue> remNode)
        {
            var node = remNode.Right;
            Node<TKey, TValue> parentNode = null;
            node.NodeMtx.WaitOne();

            while (node?.Left != null)
            {
                parentNode?.NodeMtx.ReleaseMutex();
                parentNode = node;
                node = node.Left;
                node.NodeMtx.WaitOne();
            }

            RemoveNode(node, node.Right);
            remNode.Key = node.Key;
            remNode.Value = node.Value;

            parentNode?.NodeMtx.ReleaseMutex();
            node.NodeMtx.ReleaseMutex();
        }

        private void RemoveNode(Node<TKey, TValue> remNode, Node<TKey, TValue> childNode)
        {
            var parentNode = remNode.Parent;
            if (parentNode == null)
            {
                //Updating _root
                _rootMtx.WaitOne();
                _root = childNode;
                if (childNode != null)
                {
                    childNode.Parent = null;
                }
                _rootMtx.ReleaseMutex();
                return;
            }

            //Updating child
            if (parentNode.Key.CompareTo(remNode.Key) > 0)
            {
                parentNode.Left = childNode;
            }
            else
            {
                parentNode.Right = childNode;
            }

            //Updating Parent
            if (childNode != null)
            {
                childNode.Parent = parentNode;
            }
        }
    }
}