﻿using System;

namespace Task05
{
    public class CoarseGrainedTree<TKey, TValue> : IBinarySearchTree<TKey, TValue>
        where TKey : IComparable
        where TValue : struct
    {
        private readonly object _treeLock = new object();

        private Node<TKey, TValue> _root;

        public bool IsEmpty()
        {
            return _root == null;
        }

        public void Insert(TKey key, TValue value)
        {
            lock (_treeLock)
            {
                var curNode = FindNode(key, out var parent);

                if (curNode == null)
                {
                    if (parent == null)
                    {
                        _root = new Node<TKey, TValue>(key, value, null);
                    }
                    else if (parent.Key.CompareTo(key) > 0)
                    {
                        parent.Left = new Node<TKey, TValue>(key, value, parent);
                    }
                    else
                    {
                        parent.Right = new Node<TKey, TValue>(key, value, parent);
                    }
                }
                else
                {
                    curNode.Value = value;
                }
            }
        }

        public TValue? Find(TKey key)
        {
            lock (_treeLock)
            {
                var curNode = FindNode(key, out _);
                return curNode?.Value;
            }
        }

        private Node<TKey, TValue> FindNode(TKey key, out Node<TKey, TValue> parent)
        {
            var curNode = _root;
            parent = null;

            while (curNode != null && !curNode.Key.Equals(key))
            {
                parent = curNode;
                var curKey = curNode.Key;
                curNode = curKey.CompareTo(key) > 0 ? curNode.Left : curNode.Right;
            }

            return curNode;
        }

        public void Remove(TKey key)
        {
            lock (_treeLock)
            {
                var remNode = FindNode(key, out _);
                if (remNode == null)
                {
                    return;
                }

                if (remNode.Left != null)
                {
                    if (remNode.Right != null)
                    {
                        var nextNode = LeftMostNode(remNode.Right);
                        Remove(nextNode.Key);
                        //Copying node instead of removed
                        remNode.Key = nextNode.Key;
                        remNode.Value = nextNode.Value;
                    }
                    else
                    {
                        //No Right child case
                        UpdateNode(remNode, remNode.Left);
                    }
                }
                else
                {
                    UpdateNode(remNode, remNode.Right);
                }
            }
        }

        private Node<TKey, TValue> LeftMostNode(Node<TKey, TValue> node)
        {
            if (node?.Left == null)
            {
                return node;
            }

            return LeftMostNode(node.Left);
        }

        private Node<TKey, TValue> RightMostNode(Node<TKey, TValue> node)
        {
            if (node?.Right == null)
            {
                return node;
            }

            return RightMostNode(node.Right);
        }

        private void UpdateNode(Node<TKey, TValue> node, Node<TKey, TValue> newNode)
        {
            if (node.Parent == null)
            {
                //Updating _root
                _root = newNode;
                if (newNode != null)
                {
                    newNode.Parent = null;
                }

                return;
            }

            //Updating child
            var parentNode = node.Parent;
            if (parentNode.Key.CompareTo(node.Key) > 0)
            {
                parentNode.Left = newNode;
            }
            else
            {
                parentNode.Right = newNode;
            }

            //Updating Parent
            if (newNode != null)
            {
                newNode.Parent = parentNode;
            }
        }
    }
}