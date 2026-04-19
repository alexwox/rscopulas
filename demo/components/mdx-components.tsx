import type { ReactNode } from "react";

import Link from "next/link";

type CalloutProps = {
  children: ReactNode;
};

type CardGroupProps = {
  children: ReactNode;
  cols?: number;
};

type CardProps = {
  children: ReactNode;
  href?: string;
  title?: string;
};

type FieldProps = {
  children: ReactNode;
  body?: string;
  name?: string;
  type?: string;
  default?: string;
  required?: boolean;
};

type StepProps = {
  children: ReactNode;
  title?: string;
};

type TabProps = {
  children: ReactNode;
  title?: string;
};

type AccordionProps = {
  children: ReactNode;
  title?: string;
};

function SmartLink({
  href,
  children,
  className,
}: {
  href: string;
  children: ReactNode;
  className?: string;
}) {
  if (href.startsWith("/")) {
    return (
      <Link href={href} className={className}>
        {children}
      </Link>
    );
  }

  return (
    <a href={href} className={className} target="_blank" rel="noreferrer">
      {children}
    </a>
  );
}

function Callout({
  tone,
  title,
  children,
}: CalloutProps & { tone: string; title: string }) {
  return (
    <div className={`callout ${tone}`}>
      <strong>{title}</strong>
      <div>{children}</div>
    </div>
  );
}

function CardGroup({ children, cols = 2 }: CardGroupProps) {
  return (
    <div
      className="card-grid"
      style={{
        gridTemplateColumns: `repeat(${Math.max(1, Math.min(cols, 3))}, minmax(0, 1fr))`,
      }}
    >
      {children}
    </div>
  );
}

function Card({ children, href, title }: CardProps) {
  const content = (
    <>
      {title ? <h3>{title}</h3> : null}
      <div>{children}</div>
    </>
  );

  if (!href) {
    return <div className="card">{content}</div>;
  }

  return (
    <SmartLink href={href} className="card">
      {content}
    </SmartLink>
  );
}

function Field({
  kind,
  label,
  type,
  defaultValue,
  required,
  children,
}: {
  kind: string;
  label?: string;
  type?: string;
  defaultValue?: string;
  required?: boolean;
  children: ReactNode;
}) {
  return (
    <div className="field-card">
      <div className="field-header">
        <strong>{label ?? "Field"}</strong>
        <div className="field-meta">
          {type ? <code>{type}</code> : null}
          {defaultValue ? <span>default: {defaultValue}</span> : null}
          {required ? <span>required</span> : null}
        </div>
      </div>
      <div>{children}</div>
      <div className="field-kind">{kind}</div>
    </div>
  );
}

function ParamField({ children, body, type, default: defaultValue, required }: FieldProps) {
  return (
    <Field
      kind="Parameter"
      label={body}
      type={type}
      defaultValue={defaultValue}
      required={required}
    >
      {children}
    </Field>
  );
}

function ResponseField({
  children,
  name,
  type,
  default: defaultValue,
  required,
}: FieldProps) {
  return (
    <Field
      kind="Response"
      label={name}
      type={type}
      defaultValue={defaultValue}
      required={required}
    >
      {children}
    </Field>
  );
}

function Steps({ children }: { children: ReactNode }) {
  return <ol className="steps">{children}</ol>;
}

function Step({ children, title }: StepProps) {
  return (
    <li className="step">
      {title ? <strong>{title}</strong> : null}
      <div>{children}</div>
    </li>
  );
}

function Tabs({ children }: { children: ReactNode }) {
  return <div className="tabs-stack">{children}</div>;
}

function Tab({ children, title }: TabProps) {
  return (
    <section className="tab-panel">
      {title ? <h3>{title}</h3> : null}
      {children}
    </section>
  );
}

function AccordionGroup({ children }: { children: ReactNode }) {
  return <div className="accordion-group">{children}</div>;
}

function Accordion({ children, title }: AccordionProps) {
  return (
    <details className="accordion">
      <summary>{title ?? "Details"}</summary>
      <div>{children}</div>
    </details>
  );
}

function CodeGroup({ children }: { children: ReactNode }) {
  return <div className="code-group">{children}</div>;
}

export const mdxComponents = {
  a: ({
    href,
    children,
    ...props
  }: {
    href?: string;
    children: ReactNode;
    className?: string;
  }) => {
    if (!href) {
      return <a {...props}>{children}</a>;
    }

    return (
      <SmartLink href={href} className={props.className}>
        {children}
      </SmartLink>
    );
  },
  CardGroup,
  Card,
  CodeGroup,
  Note: ({ children }: CalloutProps) => (
    <Callout tone="note" title="Note">
      {children}
    </Callout>
  ),
  Info: ({ children }: CalloutProps) => (
    <Callout tone="info" title="Info">
      {children}
    </Callout>
  ),
  Tip: ({ children }: CalloutProps) => (
    <Callout tone="tip" title="Tip">
      {children}
    </Callout>
  ),
  Warning: ({ children }: CalloutProps) => (
    <Callout tone="warning" title="Warning">
      {children}
    </Callout>
  ),
  Tabs,
  Tab,
  ParamField,
  ResponseField,
  AccordionGroup,
  Accordion,
  Steps,
  Step,
};
